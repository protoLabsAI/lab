# SPDX-License-Identifier: Apache-2.0
"""Kaggriculture tournament agent.

The policy combines a replay-derived opening with an observation-only task
scheduler.  It is deliberately self-contained because Kaggle executes this
file as the submission artifact.

Public source: Kaito Fukami, "40/53 Top-10 Future Holdout | v18 Closed Loop"
https://www.kaggle.com/code/kaitofukami/40-53-top-10-future-holdout-v18-closed-loop?scriptVersionId=340030138

Licensed under Apache-2.0:
https://www.apache.org/licenses/LICENSE-2.0

The embedded 719-action expert schedules reproduce these public episodes:
Dmitry Larko / dmitrylarko (89793681), Lucien de Rubempre /
lucienderubempre (89797849), manual player / prospal (89800260), and mohit /
mohitvarred (89805100).  Kaito Fukami authored the public v18 selection,
feature, gating, and closed-loop implementation that combines them.

The exact public ``main.py`` payload is 305,265 bytes with SHA-256
``2c135045702d8ab8affbbec441c1920fd8e5bc65eec3aca6cc2ac72c33cb9bfe``.
The source notebook explicitly publishes a copy contract and reconstructs that
artifact without a private dataset or internet access.  No authorship claim is made
over the underlying v18 policy.

C17 modifications, dated 2026-08-04, add end-to-end safety guards.  Scheduled hand actions are clipped or
padded with ``PASS`` to match the observed live hand count.  At the final
executable step only, otherwise-uncovered sellable shed inventory is appended
to available market-order slots, including stock that an already-scheduled
``DROP`` or legal shed-access ``PLACE`` deposits before market execution.
Farmer and hand actions and all v18 expert schedules, features, priors, and
gates are otherwise unchanged.

C18 modification, dated 2026-08-04, adds a one-argument entry point under a
unique final global name so Kaggle's raw-source loader selects the policy
callable rather than its terminal shed-projection helper.  The action policy
itself is unchanged.
"""

from __future__ import annotations

import base64
import json
import math
import zlib


CROPS = {
    "WHEAT": {"seed": 10, "first": 2, "max_day": 4, "max_yield": 6, "ongoing": False, "last_plant": 24},
    "CARROT": {"seed": 20, "first": 2, "max_day": 3, "max_yield": 4, "ongoing": False, "last_plant": 25},
    "TOMATO": {"seed": 50, "first": 8, "max_day": 8, "max_yield": 4, "ongoing": True, "last_plant": 17},
    "STRAWBERRY": {"seed": 100, "first": 10, "max_day": 10, "max_yield": 4, "ongoing": True, "last_plant": 14},
    "MELON": {"seed": 80, "first": 10, "max_day": 12, "max_yield": 6, "ongoing": False, "last_plant": 16},
}

ANIMALS = {
    "COW": {"cost": 400, "product": "MILK"},
    "SHEEP": {"cost": 500, "product": "WOOL"},
}

SELLABLE = ("MILK", "WOOL", "MELON", "STRAWBERRY", "CARROT", "TOMATO", "EGG")
MAX_ORDERS = 10

# The opening deliberately mixes quick wheat cash-flow with a smaller melon
# position.  A previous melon-heavy opening could sit at zero cash for twelve
# days and lose all livestock after a one-dollar within-turn price move.
# Sites are ordered by expansion phase: four initial, five in NE, five in SW.
ANIMAL_SITES = (
    (4, 2), (4, 3), (3, 4), (4, 4),
    (6, 2), (5, 3), (7, 3), (5, 4), (7, 4),
    (3, 5), (4, 5), (3, 6), (4, 6), (4, 7),
)

DEFAULT_STRATEGY = {
    "hands": 13,
    "cows": 8,
    "sheep": 6,
    "strawberries": 34,
    "strawberry_last_plant": 18,
    # None sells fertilizer; 1.0/1.5 require that multiple of its sale value.
    "fertilizer_roi": 1.5,
    "opening_wheat": 10,
    "opening_melons": 9,
    "opening_carrots": 2,
    "opening_animals": 2,
    "opening_cows": None,
    "opening_sheep": None,
    "crop_transition_day": 5,
    "strawberry_activation_day": 4,
    "strawberry_staging": False,
    "opening_melon_day0_cap": None,
    "opening_melon_early_cap": None,
    "top_hire_ramp": False,
    "land_ne_day": 5,
    "land_sw_day": 10,
    "animal_nw_day": 4,
    "animal_ne_day": 8,
    "animal_sw_day": 12,
    "feed_days_buffer": 1,
    "ongoing_harvest_threshold": 3,
    "drop_load_threshold": 30,
    "price_adaptive_animals": False,
    "animal_price_sensitivity": 2.0,
    "zoned_workers": False,
    "use_fixed_schedule": True,
    "fixed_schedule_version": "v18",
    "v11_radiant_player": 0,
    "v11_radiant_variant": "adaptive",
    "v11_route_step": 109,
    "v11_alpha_milk_price": 193,
    "v11_radiant_market_interference": False,
    "v12_late_market_mode": "price",
    "v12_market_interference": False,
    # v13 keeps one coherent production route and gates only a compatible
    # market-order expert.  The gate is evaluated daily and held for at least
    # one day so borderline public portfolios cannot cause turn-level churn.
    "v13_market_adaptation": True,
    "v13_gate_confidence": 0.70,
    "v13_gate_exposure_scale": 6.0,
    "v13_gate_concentration": 0.50,
    "v13_gate_lock_steps": 24,
    "v13_interference_min_exposure": 2.0,
    # v14 keeps the same validated route and daily hysteresis, but makes the
    # gate concentration price-aware.  A product at the $1 floor is weak
    # collision evidence; a high-value visible pipeline deserves more weight.
    "v14_market_adaptation": True,
    "v14_gate_confidence": 0.70,
    "v14_gate_exposure_scale": 6.0,
    "v14_gate_concentration": 0.50,
    "v14_gate_lock_steps": 24,
    "v14_interference_min_exposure": 2.0,
    # v15 treats price as corroborating evidence rather than replacing the
    # validated physical-exposure gate.  This prevents a high-price but thin
    # public portfolio from activating the collision expert by itself.
    "v15_market_adaptation": True,
    "v15_gate_confidence": 0.70,
    "v15_gate_exposure_scale": 6.0,
    "v15_gate_concentration": 0.50,
    "v15_gate_lock_steps": 24,
    "v15_interference_min_exposure": 2.0,
    # v16 uses a train-modal position ensemble selected as two complete routes,
    # then adds two public-state collision lanes.  Routes never switch during
    # a game; the market lane changes only at a daily boundary and is locked
    # for two days, making the executor difficult to steer with one-turn bait.
    "v16_market_adaptation": True,
    "v16_gate_confidence": 0.70,
    "v16_gate_exposure_scale": 6.0,
    "v16_gate_concentration": 0.50,
    "v16_gate_lock_steps": 48,
    "v16_interference_min_exposure": 2.0,
    "v16_gate_price_floor_ratio": 0.50,
    "v16_value_lane_margin": 0.05,
    # v17 replaces the hand-set lane thresholds with a frozen 53 -> 16 -> 1
    # pairwise neural ranker trained offline on public observations.  It can
    # only permute existing non-WHEAT SELL orders in their original free
    # slots; WHEAT, purchases, hiring, farmer, and hand actions are immutable.
    "v17_market_ranker": True,
    # This is a validation-selected safety abstention, not the decision rule:
    # the continuous MLP probability remains the ranking signal.  A 0.95
    # confidence floor was selected on the chronological calibration replay
    # split; it retained 29/30 wins on the expanded later-time holdout.
    "v17_rank_min_confidence": 0.95,
    # v18 learns from complete-game wins rather than action labels.  A
    # seat-specific board route remains coherent for the whole game; at day
    # boundaries a state-distance gate selects one complete market expert,
    # including SELL timing/quantity, BUY, and HIRE decisions.
    "v18_closed_loop_market": True,
    "v18_closed_loop_board": True,
    # Observation-gated overlays keep the validated movement executor intact
    # while adapting purchases to the opponent's public farm.  They are kept
    # independently switchable so every source of alpha can be ablated.
    "fixed_board_adaptation": False,
    "adaptive_animal_mode": "mirror",
    "adaptive_animal_min_day": 2,
    "adaptive_animal_max_day": 14,
    "adaptive_animal_min_herd": 4,
    "adaptive_animal_lead": 2,
    "adaptive_animal_target_share": 0.72,
    "adaptive_tempo_cow": False,
    "adaptive_tempo_animal_lead": 1,
    "adaptive_tempo_land_lead": 1,
    "adaptive_capital_priority": False,
    "adaptive_capital_max_day": 12,
    "adaptive_capital_animal_lead": 2,
    "adaptive_capital_land_lead": 1,
    # v10 uses the validated venks balanced executor and changes only the
    # order of non-wheat sales. Wheat buy/sell order is part of the executor's
    # cash-flow cycle and must remain untouched.
    # Earlier sales move the shared price before a later opponent sale, so the
    # overlay prioritizes products visible in the opponent's farm pipeline.
    "market_interference": True,
    "interference_sell_first": True,
    "interference_targeted_sort": False,
    "interference_preserve_wheat_order": True,
    "interference_collision_only": False,
    "interference_min_exposure": 0.5,
    # Extra wheat purchases are intentionally disabled by default.  They are
    # exposed for ablation, but only a very constrained one-unit action can
    # pass the safety gate because feed-price attacks can damage our own v8
    # cash-flow more than the opponent's.
    "interference_wheat_squeeze": False,
    "interference_wheat_units": 1,
    "interference_wheat_price_cap": 30,
    "interference_wheat_min_opponent_animals": 10,
    "interference_wheat_min_cash": 10000,
    "cash_reserve": 150,
    "price_buffer_pct": 5,
    "animal_daily_cap": 3,
    "wheat_rush_animal_cap": 1,
    # Adaptive profiles are deliberately expressed as ordinary scalar
    # settings so local searches can tune them without changing submission
    # code.  The opening remains common; public farm state changes the
    # expansion portfolio only after enough evidence has accumulated.
    "wheat_rush_cash_reserve": 150,
    "livestock_cows": 12,
    "livestock_sheep": 2,
    "livestock_strawberries": 34,
    "livestock_tomatoes": 0,
    "livestock_cash_reserve": 150,
    "livestock_animal_cap": 3,
    "premium_cows": 8,
    "premium_sheep": 6,
    "premium_strawberries": 34,
    "premium_tomatoes": 0,
    "premium_cash_reserve": 250,
    "premium_animal_cap": 3,
    "early_liquidity_floor": 0,
    # Soft-gated experts.  Strong public evidence reaches the proven v5
    # counter target; ambiguous evidence interpolates with the v3-like base.
    "cow_expert_cows": 12,
    "cow_expert_sheep": 2,
    "sheep_expert_cows": 2,
    "sheep_expert_sheep": 12,
    "rotation_evidence_threshold": 0.9,
    "force_expert": None,
}
STRATEGY = dict(DEFAULT_STRATEGY)
OPENING_CROP_PLAN = {}
_OPPONENT_STYLE = None
_EXPERT_EVIDENCE = {}
_MARKET_ANIMAL_SHARE = None
_V11_SELECTED_RADIANT_VARIANT = None
_V13_MARKET_MODE = "BASE"
_V13_MARKET_CONFIDENCE = 0.0
_V13_MARKET_LOCK_UNTIL = -1
_V14_MARKET_MODE = "BASE"
_V14_MARKET_CONFIDENCE = 0.0
_V14_MARKET_LOCK_UNTIL = -1
_V15_MARKET_MODE = "BASE"
_V15_MARKET_CONFIDENCE = 0.0
_V15_MARKET_LOCK_UNTIL = -1
_V16_MARKET_MODE = "BASE"
_V16_MARKET_CONFIDENCE = 0.0
_V16_MARKET_LOCK_UNTIL = -1
_V18_SELECTED_MARKET = {0: None, 1: None}
_V18_SELECTED_DAY = {0: None, 1: None}
_V18_SELECTED_BOARD = {0: None, 1: None}

# Exact public action sequence from venks episode 89418172. The schedule is
# observation-independent: it uses only the current step and therefore does
# not identify or special-case an opponent. The adaptive engine below remains
# available as a fallback and as the experimentation surface.
_FIXED_SCHEDULE_B85 = (
    "c-rk<%Whm*a{L#rxlmP+dU(f{YDU7aTY?@Y#)8mjV8$?Dj2CV14F9_|&El=9n~{-`dGZ#?Zmv`y#k%KxGvh==e*NFGfB)szfBgN|"
    "vw!(|_TlQ&r?YS8XaDibfBo&hzyIL-$AA3t>wo_Jf4+bIdiM6+$L;su9(?%W%U^%K`sv+|SJ!9fXRqJioSkp(e*9^>ee?Z?KW?wz"
    "|8#c#{M)zxJ3oBV4`1G0zxnz5^FDw1?YpK&em%R{e){~|xBvA1<L9?$-;Nvc&+q>5@%@`GuRnkP`kS}gtMC7=p83uC^me=b@cqBc"
    "TkzrC>%V;YaM$kDQ4^-$O@F*^&iP#nkK6TWdwsoc(E6?!rm5?FOXm%`zIwggwFeK#e9-#Z<c-b#PY>F|n(^uVm))oN_%=1|Pybw3"
    "N6pzk+&0Wi^4GK1SHF#Yyr}G`%y!4&qtU0^|6l9(Dt`F(>Uf`=&zF$Wf|Wk<caJ&TC%fm}Zj)ULe)>G^`~7%(+qpa*b{$L)y1Dwi"
    "{fO6P(@@;4@`I=E7yh(e!Cu&ZVbew>OVUn-S)|>&0FAsf?cp<~K?S3GyA*o*Vh4>4qp+%mw)lQLmtWKN>2yhtyl+-x%lKel9iDeh"
    "<FM;%u)g`~D-Xo(3{xlJ_7za;_-6j>-TMSS8*Gc)DO27tn~{ZZ(En-h6VFccpFR6vBu>=t>Fr*6bf()R)~ojJ+$0O(+q)3;Z{p|h"
    "Kz524UcbM--oF0)^PjdKKfk+v_iuNHE^U*($!o2f$M#N?i4Nk9D1SdV!6>n&6g%uZX}=YwHrW#}*Gc@KY4g4L`2GXjg>Wexd<)4p"
    "_saqvHPa)f?Vf(N61uy`N_5cJ<bIXO(Na%~yHWh&dX~}C6OcfJuh^a*fhW5!)>M~R+%)pN;8oGmUK{RA$Lt~l&!YR0evn+L`D>q="
    "`|z7v4CXF=d(WV^*l9w!i$6*zM}xZMLpeM+bBGQP({PWYrvp5rBbeJa03oK?9kq&L^eKLmfM6Ojf_MiZLeudu|4~EW2H=PN<Borl"
    "3*01ZA?{Emgrksx-YF1CxpU9I*gty9F`YYv!Kj8ootu@U#?DR({a(GnHmYM^%rmQMTvvUf{>9Tl_DH_h=CrdX<x^VnH2Z-*?ciMe"
    "^!ek}%@5m;AOE`G6HoNDBaz@!#P<Z=b%S>Y|KUX7opvW*jQ6G^4a{#NOvfUWY`C{hY-I4Xk!TMj&AX`Y;K*Gr;sf2<$I&AJpZ#J#"
    "xNJh}oPjd~3-3$w@g9vI6W`3ja4>j@9ScG}KcncK9tp;o&ZCi-01Vdjb&2pe_r*iOpN%CuFH?la^~A8`dq?JzeC>j0pUE$uVm~eR"
    "J3$aQ$jVwK9Bf4bvw`7qkf0DEE)pvyA9s?daWE^E-s@fv*%nP!dRGY~o!|>G#0p)dce9*ey3EBe3!9aJ4x`IA@BXYtfJIm_h?d7X"
    "1n!VY{MR~N%EjcqF<!VadPSLJ=N0;p<lv2UqURBrkh=c~EmLj3Bn()Wic{AQD^5LkUyZA$81@ev!%j~+eF*E~kBdaGVE}Zha5nee"
    ");H}4CRz<=K@T)^^!{Md;c#~P8WcO<LpYo7o`#jC5|dNUkQBeB6&T}EkKbu%*7F1sn=Sr;g@ViJ^`d<L0tj)XTfZZw<sUbu73i3!"
    "iD|RM%+pdd-(G$ESB0+H+Ibe?rXTI8ZP0~&QpB95aQe|Bde!be)xre+I@i;qTsOeYWWviMf9wU9j9>)!ht2!<*T9}czjr5cfH_Vp"
    "bsoB^tTRV64ZG3eM*9m5ne#r$=%>w^k(A{LMbQ3}H3P9fnKdI7DI?F5>HEMA$SeYCRkMyS2rTYv0_iGS+n&mPu%1eu{HEe(#N;v8"
    "PY0})Fy<{%TS(-rKY+OcfJZ4$7)YbTWZ-nz@Wx`p&rcLbKOYT3U2CFboP4odoE!qI>}1cRBvmlBMb~7#6Z(>g0_!&YQW(oD02KoL"
    "Njb_#p*9V=iLp#dU!Q4)AApxr3=o1-@Utlf7jla1{rQoj7>XZlz+38!J^d8izbB)xQH^ijwMvGo>=g>oThM}m<OxtN3~_zmh~2>O"
    "+beB55``D&v@1O7B26a%<*~Iz6m!Czl@vQm;<;q^Dl$6uh=J)N&yEQ#QevPe_7-}85<CydVpwGJ+`*nVk1l3DGE+SOCm%}SeD^Aw"
    "4%;EF!qoB~-(CM@H;lcJG=PBgVMTVkNg#ZB@f<#NZVV7b{JGnY1Z)Lj<Y;0EvLw%@@J^k^ZQq_NjBVLp(KZR;ns&tl8$kb7x@Osp"
    "+4a>=Z_d6|@R<FpB$4GNmcI|!UY0Nx&;^)z%8|=uyaN;n)wYz_!E!&rg!k6;Fk<<9I291~mWN8P%48<g2bogfE3PELd2>nDT9)q|"
    "HWEZ$5|lFFz8O)s!D{f6?!|mMX;N?fu@PL~JCJs^r{MuFRJsxbgW=7{+e@EeHjv7zHb0CSPmc|4aP^~(<2r#}P#FceDXQ@>HVtZJ"
    ">*c(kV*iq(Hj<lfW9k5lnMyE8N4wYzqZK#DRVys}&GH_?FUTVxO-am6tH8h4niq-Il&(}@VoM44LVsQ;MM&DHRIm#$mH@NU5reZ?"
    "t7Y%7OX7uzIDZRYN%0=oTv69nR3>EAIS(Kt*rsPj=esprfWeCdAZsaX5B?vjTo2)T*Lf46zbxE7Nq%~)m0+`$_%G8_C%}a<{B>qy"
    "2S1J&8B;GkdOW!oVof?~`o5#u5U3Z4<TCL_V0sZS>Ezyyioxql90LA<X@-qpyr-7)oDHH4VbKu+R$w@}`pR4<2GzEgO*$Ie1XqnD"
    "?aa>xJ&S)Dpt37@C-sPeH-~s~Go%G_<I2X4Qs{i!g=ULTZFX5big}y6tR^*zA^4pjcr968lB?5)(F$t;O<{rGnCBPnm26XrJu8h_"
    "SN_zqY^7wU+^)^T&9`5sE|izew-0}4aVxCnsZ8U`MRIK2DXMHru-Gap&{OerbS9y}vYCROV!jOeJfc%ujb1dfyJvBJI^Yb?O<@}D"
    "Xx!<z#-USQ)FR4*Vt@sF55G7ZjzOof1Ti@W;P!8F_*MwRDqI+s&kenIqlub5#Crx3WjBPV=@5ln<xy_@?5qY*fF~n6w`p=SV_4$y"
    "Qn6S8Vp-R(-e3?wQ6bRenP8t#iGnhb3YA#%5mCU$b6(KowpX|_P91w%xmLT4J)8$i&Q^>NrpBFngMTb1-OS?a-QQn+tAWGl;5zIS"
    ";t3mv=CDKqW|K<XLsr=q5Ug<9(Z&@?F+LShsL3<u6H#o9iYVF9<Fb~jcL#KYX`AfY;0L5~gpffbs$hV&fNp^=u%{_&`Qi2z76dIm"
    "zu#d;^1_YNcJLOn2ah4th2#Hn7`?db{7cJASY6_CZgI8%ev4f`N^Zbv^FYB0ux@?+<htQby03~f2VXo8AbrtJXP*lmI%ki!P$v?A"
    "-6vm?U3AVANHr?<;hIBa8OdR}g*%if0m|aA$U3-XwO4SA2#P{#arKQ)0t$ldJ9Up%+JG@yF7jSTDsF)2({eGb>K8}Ji2J`)-7VBp"
    "h{H6L+f2n1EhG$eu$f$fMe!8$m_^LvnsvJsX}M0q+@<5KlT#Dcn_xWAFj|S8!i<sP_jqf9_A5hA6=^g-Et!bfMGO3mw#u4~pr`@h"
    "izcC#xMNN>KkHOn*}9d^R9r0?BoJgJ5O#1wGI&MIRCug&ghdyZFAHFCkEQ`Sa3Q4DcNHqgb-(6`;o`&{vruzHb+)CQ9b~FU%0cHZ"
    "rW$%Tw^Ir{(uiH3T4B#6q|N9$!xOB_L%@NTw-omoY}r~tc<!FJaa%5GAjUniUM`-{qD9am%p*?X7FD=t3EYU;!yTC<RPZDJ7W~8%"
    "8L%*I)~tC_j6*9bNhV@#ARvjbPbd_Sw9t~NG7$p||32v|SCQ6B_OTWril#y&h0uPbvM2jW_Ju%ePkX`Js_BF(NfjIh7*GM&C6z4#"
    ";zoWEtuwCM8sK)mkNj=gMO*KjgCP3KF|!D_Z6EdF7%wDg!A}WSXeE-$b*WDG@kC(VQ8;nLI+U&)p$+yp;@Cp0aAG-2k@?32?;7xp"
    "@CllIWGsJbMM)!0$HB{ew?~vM!YFfYOj-1Br}aO<D}y1;q5g)cHU}tagze#`rJTc+I>NdRF0gD+<BpdSb8<>3qlUUd!W2RJiI*1y"
    "SHek67;GlaWa28XAEm04PLkm*1nzATKCg_Rs`ikg`vq=O3F6!Jh#P|n-qH$08L3Y-lcWQ760qi2hB&o16?BDWp<d}l4Lw^g<;-&F"
    "w)Ewr#MAtdfBx{`v^o!{W-ImbL<rl7nOO^RIf5Bt;*C9)W#K5sCK^N9L?TuMwCUWks)0(~u3?d?${k%|aB25)B&-I4<uRFO6)a7Y"
    "p`6k4qI)(|h`=%5TotI4_la5)ZUxPf4kqg?yKXvF)JA?D9#0`JYoqyg7M-Dpagip=6Xv*DlRDZFl;I^3*v1NnaK@iKaj@`Bv%MOq"
    "3_(0zxD!r<UFm2J_t7kuW2D|U;Qm7t0^xq=K<IOgP)Oz`;Z;NPWt5IeRzV$-UWk|x7}3&4SP9ssSy50>Zac^9aWsb;KF3Y=Y&v2F"
    "+B)@7>yXB_m0+5L%|5Le&q!FHBzFmb835YBHHKf|R_Q?qJ$7M*i38#8ax+?Y)Bh9tDKEP0x$vw@PIw559TP#%m*JLq8cu#eb<fN)"
    "9zADklHUuLEMbn6fc*lakfG_@EZcYP0i_jY$<xZpsZr#tq;fdySq{7Hr+&RXuvmx1-@uh3ZhSz&Y9qsi!>%g9SYDMt`1+v|i$zfc"
    "0L&0h)>_Q4oI+XFE*dnRLK-1w$+9h%#0qm5FO_tO<?e-?a<qHoAVR^iF4FaI$b|cs*mAq#sQQK`w;HFclADKR*cf9GTyS`!nhe#I"
    "EZ6prLE;Gf)3*pPRG(FjBbUmr%%$>WCMN`3T*BqnIO22+nn5}dKBq?I0k+i%{*d9OV=EYx0g_<elE*Su0LB4B3o2Af)b*?E>Wq|E"
    "DqBm(mXfS04#%uKxHi7bF(`p<)C7wb`=ZU4Ifmd^<e&|4be>~qBc|Zi_nQ`zMoL*R*uf=9kGtUM`7SG6OTw5`1bjC`x9yD9oFY-Y"
    "9G58W7Xz`cGHrP>vHIQ05;=N=qR|H^ON@{rsL&&Q(|dkDtI$SKl;FX9AUjtbOZr2#Pc(;p<+jVLC?+@&Z_Saj)xGglEdL3BNGkH@"
    "j!DC6Q*lz}MG{Eh_HXF|3F_k8s8JWI-4kvd68*{y!deG_&$&><Das}?I81u2bBAE6X97p-i5Z&Wk-{`@6$;4!zbNbNh0U7L+Mq~T"
    "Q@txTqEsvy3W$RmNp)oCaM<KgJpeg&SL@-M+f((<_DIg$OEb9Mx+~b^T6j(H5Tn3&7@QuLzsqhBB;(}P>H-vRK}y{a0`?`j`<W~`"
    "QziTxEUuk20%zV^9TUNP1r^V1uyz<y)iK_z071cS6us;~fwSGwb>m+TCQ7XX+?TypOSNG0XxD;lq+4w0wgMV>!i576xnAf`pwUAE"
    "jW~s>O?&5G-#h@y(e%S(;j(LFmFFPw7`;Cy)=R&OH*CJG&p1H@sOSPiEfxZm)Mlxot<=~^FF+@bcWez2lZy6(2MIcXBNgzlvcqnx"
    "y&a$@d3|5SYYRyaO@hVCO@r93i`Zq~X>=V41Ue00aRX2?+`6llCU?{w8YuIwtO6xTJR3w&8pM}W^0*;QWhlO?ZDVK70JnfiM`Q{t"
    "(a@q{cZ~nS?K)TY>l46A_R&J?TiujuK|PgXMaI&Z<PCjjpHOrehZhXF;a829X7RY&lZ`@hf+B3d><Eq+iP0r8Cyx8K(PxU;79tZH"
    "U{EjZaV}l2P1ePz9Ff}XGZc(Qv%DVC$jltVT9_-!<_D<6w($$RHZ_LKa`(NmhL_GwmcS#-%%wgv)MXA8kGX4PIBOaWjxmjdnK_*e"
    "l*z&Y!t}gP2lto+Z-d3?9~=B68#+B~qLPlH9WFHimGb|K;^Y!KJ_V6-Mcg-RV3lOz6CbD^-3%wtfg7dZF4jU95!~2@9*&z>MXA47"
    "-F|tyC(%JiGP23fPY>^d;B$c30uYQaMGq%@aOshYN>=SrLyN)?yMwjGNpkqwUd&(Rs6*EUInF4uP78(JPKMT^SXA_8+G3z73syMA"
    "BTa589g}Wa0hAlsCp_*eK#Pvr%6vkN$Qak!s^DJ&{2irBaGSdDSk4EG1uE)>a+k21r&m_WZgpmWEb(52nL5o?PE@xtes_`f`_cgT"
    "_yl3fLeWj0>Y3>Rk5{gp{Pf^?=O6XN%24s3M@|NL5nbOXSIJ^F33WSL1k9Z+xIROrIXqI@fJppR7su$3Gi2S!BM+n}cF}``ces+Q"
    "t5BZ%PB|^i1HqdSo>6dd$}UqH1!@#>Oqw5v{E9{cZ@vNv^@tWsp+&q~4WA=wh-|X0VHtxEM%DQ_kaZ!I<Cvqu=&oRbIYDp=@h3*A"
    "@x8LT1&tQhx>1e1cpNfJ%<BOGLHGtCS<!Jn3>W*>90l|2Y)R(yq<Y<wD$|57(hxcYPUmT|`;}`<KO)vqKp7G<G#+)MnzD(wnw)Fj"
    "WoPyHzd9q1r41GE>iBWU8jk`IR7Y464KhXvmB6apCPJR63v{&daqQ87qR?AQY*BZZy+_<s$G%<Lr+zuV9$lA62U8@|+@5I+QkDd1"
    "s2y}2Q3nN{?I~vZR_mqFC4ob}%qu4>Y*Gk~BO*e_6dGs5pW?46jW^Xx9<l><WYhhOj^Gn^^vAF>;^5IEu`?2|BONbN&9MjX_Xn5c"
    "4bj9?R#O*tl(sesy<PLL&521RQLZamP|;TggS3`JHVDQC@z(^fgp8;bLTa4**J$ZQzlS%-dYXP3>_((*Y35Hf5$nST6{%2&XapIe"
    "Zlw>%2#htRaZ+@xS6PKCl_ZNQKS{<%$6z3UJVx_=Do0|-6*o7E;2>I#E<$di3@k8x<C(FNsgZi{jDmrWb9mMPXWiNF2z<3#3!DQ7"
    "rlOg9Kun^ngaSeszyp#4H2o!*kko$C?1oS`@KRtku;TWCNd^}~U}QM10SFsfQBOxgQ}Nic1~*`YkZ&CEbH4qZV0E0pmZnjre$&(q"
    "4*lh^`YmM_#58nL-Mg%yEnET5hjEr8RAnupDlhL>b9MQoxvFWdIAP*<Ve<o%cpqSad<%9DQk?t_<&jc%E4amHmkWU*&`FX&zVKKy"
    "$mCYwM{PJ5ryfP_E{9XfN$GKkDg$$zgh^6zqjIm+m^x?yx(EeGn9zz}7DfnQD~T)wC^;vF;sJ4ur2a`^WEVc{k)fDlD$|DLtuns6"
    "G8h|JpjRxQWQh=)3kDm9OVJ|$PDykR$02Ny4~f|Bu@YRK4M(iQAmE2s&1<qjm87kd3N{8P1{F9d*5D^`J%kbX#tkad^ZJd+p&jse"
    "5ut?u1B(?rB^q4`&BJ(2kU%p;H3vPha_j7Vq^=-qSsD;_+%0RsT1lF#v$DmZ<J4;Hx(FwTXaEn{_lr2}=Y)!a#_W86nMUgFS^TWv"
    "h)wrS2p7F*qUEfQ=r_Ci@)kVuO5Gp**QGsv##lTTPb{9L{iRi-pa*UBW1ywB->h^Q&*CM`G(T#(dSOE2vws%xpq61W%4|_fW~T(g"
    "y7dC3=_p`fC@ZO`!ft#sYD<@hS_ak@mACZQRs*jSwaqq>bF4$`L=bIyB_@X+;X0xS<8~0+a)m>oG>ULrc$fV~krdK?nf@izM+vxv"
    "1c^K~y6c<h)fC8v>$F1P72&HG#gb*Q;s~s(n!n{LZ6l(F=gxlSz8z9X-_+A2iGez|!Ob7Bh=wd<0k*IoW)6_xc2RdAu3HN2yj5rH"
    "b;<{q%~8nUqVQIe(f7J&g&VqzQRlAcbdgSsweI<?xy(4qxMU?_AHiXr*$mB4Y~^Enar`$7fZBiG+~5X7N&rr<VxVvmFH(mgB=vfi"
    "0|3%N?`^k6Sj`W7T@w^ivnl3ogqCEYBZ-*-N1z)YNIXT&#=U7)Jl)>8C{#?lYVY=kbya$$z-^ud_ucE5x$6@s#qQNMU`{SNv3i_X"
    "9V<TZxUf2OcqB+-tH*@ZL0!?Q>%-~RKs3m|+2+!?Hu0TOD9$Ml7v<`wEEab|7#f;~xgvk;FApaZ)j}bvB|piPrHAK%RGFd~ExM=`"
    "G;=)_;xJNcuQUnX7RzJvqZ?r(G0BHWplmi9@SR5Mb_=%CwkZ~`=GisPcsht?ys9`^{ThxCGUSqF*H$BbKLPMW#4@4EiV<X*04im9"
    "NmMlU+J^B4!@!zKxN#*1k-*{!6k$1f_%1O~9+%I{NJY6awNk3WRH9`}RI*MrxCwE5fb`7ARiodO!7XWG(I<0}<Z<&6sv~iX7QjWK"
    "qnjBW;o}ejT0>>*Qm_>@FuI$fNchPuk6M8%|N5T0V6jG{H&&@5Lvy0kn+t8|31HeaBPU1SZV8W<-h(fkC%*tt+lHfx_ouMjJSi*}"
    "scNVG@%@^ech|IA+~4Xn1@4&j4Cg0pRjZvsw%E}#CUqNd!dlutU1k^NT|OcBH~_<mXG*b5pFBq#m9An&A`ep&P*Lus)jIJ*pqew-"
    "0Y!GXtUOt?NgM{?ka$<t^?^EHzV<z4YO&qdEFrRjm&lHor9h=Bi9{7~X>LI`DSa0Rp)=)E!}5eaNxWy8ZWNcfgaR*Ei{UL4VtkmR"
    "Rw7So#$KS4J^NVK*hVm-jKy`hj*GV*h%YI+kAa2{d1P$|{HCIzCDvpl)7+RdWeDbsXm#PyL9X~+K#F)7ImaYGzNic~nk68TW^ahl"
    "8j50AEG`g{6G{ghYMn=Tr6a?fS6ol5R0G4(ll(?U((m2~>rnxbUi5lyy`r+o+YR#a$=m!%71_wN+B|xu6-A|&u>I|(S_PF!`i;&9"
    "xK_kTYbjDsINV{)wQbqhEWEa?N+FS7nMp|at-4eUEL=0>eNga_lgr6vSUyF^xlLB>MYu@&D4i?uWRI-^S|!!Uny0dMV^c>ts(WtP"
    "-%|Z4E68N6PF&Ovn3YMD)_h;_C@=Bo6BOEBjC^Gml3uw_OI=9yh^AiIlT9q$$M>*IpctH`b%r9x&<|P#-P<zm9e$IJ4xpdxwn%DP"
    "l=S7;R+}MFAVjMy6)Li@3Jkm?T_DIL(>)~g7KH^B+=en9_K@OH=m(srsV5xS%2@*r_C#d(?ba9;CupAY1oQ^=6Y{oNmjJ?O;5LGC"
    "c%~zObj00u?UHcotM>94SL=Gey~u-#A#eO*8Png*Rl5qJ>k+dqC4>3Oy1-c7>RB&XcS%Lb_Ru{hpQTsA;_%*V;Gdm9ig-%IZ*vN<"
    "=>7Si#lhK&6HI6u1gMwJqkzv)zWt&&_ZsL21a*>Tt=$VvB%l@jof|6%2-S!at3_hE>2#hMV$ox1MJNiIystV}1ahu(LM*7G`3~`$"
    "za{8XP%0Y<V?mlt^K<Sh9l@eK3(T&f5xhtFNqu_|E|~6wc_R_T;#}M{s1L}-)<BKHG{u5_in?S`{G8ypsFhaD1Lh=AB3BOlz9sCy"
    "ZxoIa!vO;RlDMK)x+KZ*NV6A7cs*eR%S;t5Y&awrlRWIU3e1=B#0Th30s-}>=Q6?VOIKdJ(LD4M9l6w}1u(P#PQZD5!T*!CQ`5YF"
    "Rpv_7gylw8sE@ui%?;~}N3=qo0Y?UJyEA~Byh`YpkNc4MO2R&wAvsW(S{FNZw?HEd4J_-}cL28rYd3jGl4j1e&tWgkZq0GJrHc>@"
    "OENE>B$@Xr$$Zeo2Nuy}>TWQPNtUVm>yOyt--9ZXy0zDweX7MF@yH6CFOdKPLyr!(v~=+c@QNf;1Gm=wREuN)RuxkOfrUEH{k2L7"
    "txr>J2a!-o>X=+`OrwU^k}mrVQsY7qNju_GIGt!$kSJ2Q6}gXo*}{us)^I6gRHM9vQ6(cq(aN;kbWqfx_V6i@vf3e25@0|#kO-?S"
    "43~b3B!M@INj$$O=%f#`wR|HP+tRB`0~|EBS|QfP{`P?4q}C*zoioa<5(7^B8Z(g61m0Z7R%T3f0ZLCOptc3^NxLLPv0g+93jZlO"
    "I@&gkk45fr?Q>&K1<S%}X*48X`RTcWtTF2_=q5)O)btmYedfZ;YGKm*;ETOfidZ~JIUgplv_Cn<h+P2$B^GL29%O>h(GfV97l~0W"
    "_#!$@3UB6XdxbHkz~E{CRX9Y1%G=3WHL3)w+g6Lylz<jfn$SQs%t@3Tpq}iL#(+M8H^4w_AxMwmMm*!B7VgfB)agRvnj`E>>I)*)"
    ">sqS)0@M>8;Zbf)M3^_*C0zW_F%;RBu~a(TcI>W`^nyBf30kI%T{BUq6h*eA7_CajV8!4HZ37J<5-Y+*q6D%b)Io7Li|POp0Hsn2"
    "G?l9f=-ZT3MYT-1uTrx9q$nJUa#VpW3b+v(vOz7;2Cs!PvRLL^MXNP|KXkFh5)YpcQCyR$#M?_P^<gG2X9&m(A8rLB0+C`9VP567"
    "Y35wuK5VuyQh3~~cA{z<Mzp0lSc|I+PlWEQpGlmeL3F3_3ia|toSR31?hJxD@Q2uE(~AMB=+1aIqaZnvbL;Nr%u{K+OE#Xg*XL?v"
    "rhABryQJuMq$JR=PkpXd>*BdU!(7V{!2My|AL@Fsj5<=S#!j&jn32@RIGSZ4%HodjB-B3z+l3;=%yh&1kV;SqFGj7<nA!iGHp}w*"
    "tsM9Pb^%g~43YvcM^!swbDz<n9;C0>24yUmD(XF9ZO}SRl9GjDpEcB?RJ#cgwT@XIK32_m0#oL^b>31<SQ4%HWtbHrvY3~oD_R+@"
    "FjLEcx!RcSp^ZdN8gL7K$D6AxT4oA*0LDf95WSwlT;>!>vZ^i&mhM7brdXjo%h-wIaBBkvY_EJmrn6$es*uLBR*TN`fx4`*GJEKV"
    "wv-~ZB<DDZW`0!1#F^&?TJb9D0b{WZRZDq!G^FbB02xIR*~fx#NO`D;Ow+b013>7^0jrDB>q{4IUeJU?Qpu<<(;^oAsTG?kRZ&k@"
    ";8exYWl)?syz0y0*LykbRx?Jhx6t<NriO|;R5)&$aslBxdD?d!fa+s+PQj>>-Cu=uds*ug7{a<rg66pHfLM_15|_z@f|tQ2etFs_"
    "fngYK>PR!}PaZ}q>`$f-TxC}?uC!F&V%ySYF~`o7dPB+NC=H$cpVl!X1ZwgV@2|q9y&Bq?y&q~D;;9W!Z8%jMzWqPXT4Uz"
)
_FIXED_SCHEDULE = json.loads(
    zlib.decompress(base64.b85decode(_FIXED_SCHEDULE_B85)).decode()
)

# v10 balanced executor from venks episode 89448781.  Unlike the rejected
# cow-heavy branch, this schedule improved both current-top train reward and
# temporal holdout robustness.  The v8 schedule remains embedded above for
# rollback and paired ablations.
_V10_SCHEDULE_B85 = (
    "c-rk<O>bM-a{Mn^YhjX-KeE%<e6jH~W5XX2UJPR}Kwc0acv(!c3-aIN8Bv$y?dt04K7A=^Mk^DN#e2SQcb%&0um5}Y@4x)|kH7zV_D^5WK3skJ"
    "boTB1>_2|_uYddR??3qd@gKkZ`k#OQpYNZ)p1pndar^zZ2Ooa;^0%L_etP%g)%Dr=+3WW=XXl%{AAj0z-+ce!kK60_Kb@UF|Mu<w&JUmT!<Tp0"
    "Z+`y%yw6{L|E}qgU(ar~pFaQo?LWQ$`1$SGx8p|q_T3*ozJK%O_2=(jfAe;G_5I(~Grw7%-fp)azW<kb3qHJi{nsxa?%KUNYQprp>5uo#IlpV6"
    "(5~M{<J0!~diJB+md+b=ef4^~YY!ff`JnZ+$s3#9e^?*2hc)BV`!Bms^YLwJ+FyQKS4Yj+KioFVO!C*W*H^!fecYP08I{@YID9nvbo>9MCkazx"
    "9=^Rg-Y4htC6s6Z7<tqEyT=^vlil-fx5=&rKYgC|{eHYt+j(7D2h)RYu0C%+;&s_H6nCrq;OYB?KW$gA7xrJ+v{A{Dv{PXgY4<KbBM%=HjIVvh"
    "G^k*7Z<j((U+kc<VT7w{-nVn_r6u%w8qej|w0$~V(&&BD*mJgw5BAmJdB@taa@%6YVb|ASee=~<9*Er;rcT1`E1=eKbC&GhC-B)|Tii|=e#d~2"
    "r@9ah`acbR;@OG*vu7WS#EJSnz1>TX&UBl^dez>Yn`9w;dl#bqP5c}l$W9T%>-X2!+t;6e{>%2`=Xcld{`KzArEStTd98Kx*xrdU<-mVk9h_j4"
    "SW}7}cAm7~3R9cx37G37e$ce}-h6!j0q#P$6b`<H<eU3t0gsyLk<)fhKU)di-D4#>Xl!!7%H(LNr^VeUezAHkFVz!}K!mT@o*scGyD!#Mmss31"
    "^1a|y(b8TU?o7w*A_LE&`;mT-T&ekMpPKvdn_CR#E)EjJ<G0vpLb;3k4drwcYBZFiLEZA993GrGM2ClIxX01c0iMwj%<UV15Yz0AT17GX6u)sG"
    "SP2<Hyn_&->3Ep`DD_qaK*RovkAIR52zmVJ2R^Jr^$?Ci3Z$!XB<0S1G=|>MTaM}6A<Qik8`Qa3Nowruq|oox8*HOG_QgE2s>XHIC+c539b}K>"
    "du>iTds057B~P;-=+o@^b()8L`uy?g=7;UakAGY6iJkG`A3qL!g7i)e=^gxs6G3{~O?@#6oDMoLzm4D>i(7_*Koa`tFL%^nn>6pDzJpV6xrh&R"
    "Yad6C1bp_3{ot|*t#by>3@p4a&BuE*eoTBb3&X+SCH5u=`TUHccX}ikXF7QjuW51>X}>qI<8xm;6#Us(vhy-UcwA2mOTKqxKFQZEi1wNM@+r#G"
    "qP!D?dxNa3WdgreBrqESE(Zw;!QdiMVe)Y&i4O-8Rq4I%1(9vhWTkhNK+*}m5JRlcReCqe38u?j470FV8R!^I41!rdAZ)uj9_x~Fi=3r$HO8L%"
    "A9(@C;N54Eo>%EZl7&}^cu(+%+#)E%k6xP5{{5#8RM!tHP(63wjjN|1_D>tcPER_04C_LTi-fRY1azu+Huu`rH|+>0T8(Ex4>Wl6{&3Racy{_4"
    "6g=NUJe%*H29~BWlT*(S2wzzM);d9d(8%mwW5355Tu!eS<@*<ah%4Rt15qviyfLl7$2?6;n`LI6mZSOh>f^sCeAU*@vj{i+Xisi~?(UN+<}`=X"
    "j~>yhcK4|kCh*s}o*w180j?$!W*+%tFSum{Be*|o-oL*F{v`UvJJAEofm*5a&{k!gIihLUjTSfBUuejj_en-TZPtt=El=RO`%iD){j=i=GJAL`"
    "K1Nn2_6cAEWYz$+s#)t71RySq(FjV`c1Z`xhuB2|+RDzym?h@g>43Gug7)x9><Wwo&iVtGD*$+u@`Qmb+IH#Qpkd%^@%%(#^z+de)U_r`hRGMp"
    "g~=hl$`1BSMpDIMTXap<JE1R^DDZ33FNLwh0#G5)pOk}q6lv3#n;6Ta^!1r$`0;l+!vGOD#Xg%ta3Q70$#Q0nVkmsH@ouRz_ViO=|DJ@xrkcKa"
    "*D4vUvR5boZ$S$Nawi*SS&I6;5xarmmsi?$BnmIkX;*mEMVe0F$zyAaDCUGKD=BuC#B<5+Rb+JR5d+glo*mO!q|883>@D;FC1xHHz_7^XxdT0K"
    "9$n0Q2oWD*lMf|uzI&BThwTtmVQTr0@2>y48^+#98X!RWup+zNBo02kcn+UBHwK6z{@mq90<{7$ax}37Vv=W5c&AR|wr|fB2D$97Xq$v^O}pWN"
    "4WNIkT66sNo9nBe-kg1_;4%AGNg~TlEPo%by)0oYpbIeblp~kRcn4TqlOUa-SpoV#;PgB_j95M&P6b50<)IR+GMNeWL8cV=iYrNQ-dvKkmgPH#"
    "jRcXG1f>kPZ${j3up0cNdoiC*n$%l=Yy>&?4y2v!X?Va3m97NA-gq<e_R?pV4W#m_%@3nS(_=#$T>YryxK5xKR5t-{isId|*2k=%R<>Tw`ziJ>"
    "Icg)h`8Fm$u$ZX?lXSF;%`jSVb6mATvfnK4A^d_o0<x6E+_Vb(d#!nqXie!#1tzwXZ!h%c1vyi?`U7H{!)L%)0?bZF49;e)mc7F+i5Di~{4IPX"
    "#d~0LMO|7^nUGcIJb;j3o1Ph+?$&St1}_qTs->_!_<yL9J%sCB=S_tEvT*w(`RTD%g3VeYz)Vk_02fB^*NKfC^f)3xrCxgUcyceqnsn6keMhAs"
    "P%aXwW#Wy%<RW0w$-NyFgV&ij1pEWj3>yJH`CwAF=EjDw=m-HTFq~X{Wv&y0YTL^uos-JK8y%9gGd~;jEdFVL$*$y`)FTSs9OB8%kQT^|D;qmX"
    "q4Qal%@(2B?6P_k^ETJ+CAE=aX=q08S~9mJSEmo771jcp!UDe&&oA67*`^eGRvNP|{HbTzO36;SU7LrSZ@)}kC@-6DAO6tdR#?tcnZ}ok<k-4X"
    "RN0eYu~k%{r{d}8OhSWYGX*`xd>Qn4M5eYHy=Z24&*J=az!{#I!Zh5`xYKcsL#MpDMN|jH01NmYesMS)gHB@!VsZ|^?ce0^tq_J)xG*lC8+z|X"
    "6E%B?_Y5S;ZU|A+Aqv^{mhcuks{s_S^~FQ;G`X2EEOB{Js_Qa-EAy#ey}=-WqC%j_Grc~c5(Q-<6)Lf2Bcgzhfx7IrSGY4y9eY~2R=bTooCizJ"
    "R(ueq#+`eEf2=0m%;M|a-(P;Kfy3zFI_wnU2^)vzuxbKklS<q}R@oL1tZ>`W#uZ61J{3`@oipbXQEbf@d<I94%UY`59ncY`ZL({FACSrsLI#n@"
    "g8|wCx&^+#o~Eqjhuc?J5VZLGeuo{&3pY;N!CTB8Jcdvgj{nPH^y04bFD)-&b&1cp#n}S*Eq3`RxdE%q0|hI<y7l>!>xMh&zADljeDOel^hG<J"
    "eJ*(DoITz`ok#$7pL|Jn(K%Bf)u`BqYX*^JB!}e|?og%#D2u})>)@8vUcoUUC<>{?)i*u~C<wOi)IC~h1IB2%$a^8FxB;S1%f+;+UmPVP?*CSG"
    "w@^<Z4%1L>GZjy?kTBH2W^xG@#Z%B@7BP=&*6miL<vIy-myWkiPE9y(g7HMdXeD|IGe(Nv<E;tWuM9m^q|yAe<RNAktxY(Y)lkUK;q+;3C}|RE"
    "i96<G^RrIHl&xFoOvTlbK>|Tm0$~UDBZF7OOohifM_6=m`LX~O_h=fR0~ZY_D&qk%)}jj=cT7sn9~FcI)K!m^iOye5HFRihsKm|1h+UsrV$Wry"
    "&G<UQQ<^Xx21j1rQ{3mUWvd0@xqIHmZn>zz8289~xp=~h7Qu@!lQ@Z7Q~{$UkRxUgcVv@LL6H1=@Dr0{z{0e7v*t=M4z8#)nTWT6pd`XSp<qDL"
    "L`x>iL>w^u`=qa2MO-iW$6AainhKE?Li?4<pDc&LmjbOl^#yOMrWC5ARd5(!tO$UYRK5tP8~H`F&e(2iklTTSuuZ&Z>z#8DL|?VSa%?*|F<wa1"
    "f}aw$&`Kng>r$QY<H^9f!*Jq>btqmr!W-;y#L<OV<HVAdBKMC;-ZkhOArv(G$e8}r%92K`j)Rx`ZjUHigpualn6hZ%PAh<dR|dnHLj?}gZ4Qvq"
    "2;aj^OF4@xb%b>tTwvLt#vLyu=H-;oMh$g^geijf6E81lqaIFb!eTRVCKF$I{U}w%bdnHnAv|xBaC&7FRkeo{-!HJ6N)X?!N8B1zaF<pf%1D2z"
    "*(4qClYlwLGSsQPt)MG43l&Q*YUtT|DQ}ibx1}#1C7$Mw{PTwgr`4H2HDjrlCqmdx%*|Sm%@NEHlW**?EDJ|5KG7J`CQ`8?piO6&RV`HNh7F69"
    "Rqp5#gG;-YBjGg=ERP92t6*uG4&{uV7u~a&LKKeq=BhxYyie4ea4V>mbTC=x*>w}DqBipL@OTP&VH?f4v*-;)tcx^Zo-oJNqSVojpd2re!Zubq"
    "gmeDviG!tYn(@^@Z3yD>!o6@J>`F&-xRYkN9wT+X0rwxG6bSb_2ST4~ltMB$3AY-WFQar+G7IVu^+Lpqz=)PU!fL=a&5DA8a^pE>kfV9r@HuWW"
    "Xwwlh(AKGsT8A{gtpw8~jP_~Kc*ep4HMvUw%mC01t}*-)w@ME}=&=hcOdJSzmz&YLoBp5BPkGU0&xL1Qg2F>s?3fCAz6`g_Q*rVOs(WUZ@#r~Q"
    "l>A<}WC?qu1nd`Bg$zyKX4$@T7bvYXOP*F%PK_dOC6&Wr&vMvpKlSVFfyFv3{syiTapMCDRvQ^E9ClR;#`3BJ!q*R#SS*Sn0APl2vesgU<&?^@"
    "hS8w$6w(MeOO|c9BvzQic&VsMtamTul%w4v2N4REb(yY*Lni#c#FpC?NA))}xz#vjmE1fm#Kstl;DW;&)nuryWVyC~3=&7+pT0$arTVOP9Jy3}"
    "WiFL3GdUsP;u0>m#u2Au&<xUv@Hw?A53sFH@P`aH9b3Vo43Gr#raYFh0x%93T2P@<qOxCQS7*e$QsG)Uwv=R5@i=A$!nN^bo<Rw8qoz@`*cWZS"
    "%rOMVA_r}Vr}G>`8?gnq&fm10G}6k7!457;dfWw1&v#krS`x;jGT^%zx@~8)<`jwI<+wy~zZi&tm1)b9jnyAkmdMc~6pcPWSz?3~L4_Xao8I&L"
    "S%o%=q681-1lhUjSkfP=eWE$+E4W=|MKQsNcx#Tlt?rGdV);)1L{gDIcT5^qn~IY%H<CaCw|`3)NKhBwMy<M7^`3C+kmy%t5Y{>XoX&+ZPEj_I"
    "!C}&CoqGgRJrg)uPt4F1j})ePt58S=_(fTFFKpM0)&@n&nkrtg5v5|uP(U11Nvb15hr=e1>H)~HyIK$5+@7j;wnuXIUYf=A)?L9S*TQRphZqIM"
    "!{GF|{9Sg7AXz83))%073sUNa5U?-F-Opsnj4I*ZU~%oF6*#lo>X-=TEU36<gSEqus*dqyB?t<Jqv&M^N}TPEt{eY)FyS`_a9{RXEmecfqg@NK"
    "k#4b}+Zt%#2^S7P<hr3hfkqDvG~yJhHtn5%ee(b)N7E0Fh0CsyRi1;yWAy%<STFuA-mv+$KH~%tprQ*5wO9yLQk$j9wo+pwy#$>&-mx`AOe)$B"
    "9wg`lj#R+I$_~4&_jZ7u<n?_OuPr1!Gzk_jI1OUEE@GE`r_psJ5a={~#SK8oaO<vGn%q%$XrRoyvI>+W@oW%9X%Jsh$>WAJm7(~mwvC-R1Ka{8"
    "9g!)pL_>>)-7)?Px9eQpuTKCg+DA*RZ*^0u1@%;l6&Xusk~j3BeL~S^99}TwhF>*an#JR8Pc{n435u`*vm-cSBu1CWoH*{^MxQBWTZl|-fJMEu"
    "$GLR9Hdz;=aztvl&rmQL&GL#!BQtXdYhkV^n;)PO+r}^O+SC{_%iZ_N8eTd#SpttRGne|zP?tGWKIX2G;jC#iIL0&*X6AG@P$mlp2-EXE9o%CQ"
    "ybYG4e{ArRZ0Pi`iAp+(cDU38RLcJ^ijzy|_!LCS6>;CNfmM=;Pkf+ybTgbl2X2&tyI2cdL~vspdN^)k6{Y@Sef#C@o<s*7$;c)@KRvt;g3keB"
    "3qUZ!6g`~q!KFtoDp|cp4J`^o><-ozC&}S!doh2NqYhmc<T#_qIxQ4>I~iJwVo}kXX^VlTELh<bk2JZZbWFNw4Nz`qpYXV^04+LdEAt68B4b=@"
    "tAc+C@OP9h!ENfoV>ur%7O1Ej%3Z>4o?clgyVaQivc!88X6iIoIZ@rp_}xX??@I&V;}e7_3q?11s%NGPJYKnW^3#LooqyC5D?`PD9yuB0MRa|m"
    "TqTRyB-HI}5iobQ;Q9=e=I}^q10wNPT^yrB&X9E@k35i`*hLQ#-r-8Ju0nb4JLR-64+L*Uct*j+DZ5N*6sS?iF=>7v@+%q*y!i?w)FWCjg_iMd"
    "HGGb!A+pJ~hGh&w7**%zK-Ps+j$@7rqq~9$<^;hh#Ge?c#`ntV7BpI1>qa&5;&I3@F|P*%1mPQmWJSmQFkI|ga}><8vn83+lj?O#s!S8UNJHop"
    "IGv}-?pLlc{fJmc0cA+c(0J61YRV?!YI3f9mz~w)|LTl5mNrzttK-KZYdi`>P#s}OG{_hwR06AVn+SQPF3{1+$FWBTib8KKu|?fs_8xIl9s71|"
    "pZewedURbP9ZZo-b9<&SNLdo3p?1)9L>&}(wx^isTdkKymjn*^GOwJlut^~_j)({yQ)rwKe~Q1RG~QG%dB_gbkxlnAI)YEw(VxT4h=WIu#Lh^-"
    "j&!_8HOC&j-yd9(H$)RpSxsHoQQF!l^mfg|HYX;PM7geLK}BB~4ANQ>*&rAn#9tG@5;CG%2&r-IU!$cL{T|*R>uLIFup5!KrI|m`M63@VRHQ;7"
    "q7h_>x|Kd4BQVyO#!1n&US$=oRFW*J{3ID49fN@Y@)*tgsT_$RSKQnvf`e!|x(K<6GO)n(jc3M6rbg<)GYSSi&f!@HoONfvBk<K~EpQGTn2KiZ"
    "0WpcP5()@m01rqG(DavJLQ?xlvl~L)z)OMEz>3=kCK+4|fsx_31|V!`MLit}O~qr&8r*;pLcVdt&-wOug4J;XTbf3h`b|?eIP{mt>bI0#5Yx~}"
    "b?>r*wr~YJAI4dZP?fcWs=T~k&DG_T=BlQ-;)IDmgv}33;(dSx@-5gsNOAHzlt)V4t>6})T`mNMKqpB8`NCt-Ad_2xAGP6NoO%?wyBtm_C#A<F"
    "stn9=5++H>jmo`NW9py<=pqy#VL~f@Sr{RJtt7G#pyZqwiU-6slKLlwkzM$(M}}gKsZ1M|x61hP%3y3@fnKqIk|jcHE*NYaE=7+3I3>|N9EY$)"
    "J|tqh$4YQ{HXN}IgMc4mHLuAARg$(+D%cpH7*yb-Sc9L$^$<qj8#ky-&+9iPhjzf@MT8at3@ldglxTD%G!Nr7K?2PX)g1K1%B{2ek-CDcWobaz"
    "aks1iYb9x}&dL^tj#I0(>mr;Wq5(W)-!I~@pA#wy8ng2OW*VuxXYsRwBR1VTAzbvLiI%fIqTlT5%UkfsD|LVLUzhgy8DsHWJh6C|_Lo+Tf*!Qh"
    "kAarjezVeLJd2k!)BLFE>V*l7&;D7&gIb2kD6>T^nVk{{>(&dDrlWv`p{%5$3cK;ms4ZP0Y8hBtRNm5GTMfKU)Hd5h&an=$6G61;m6#lQgzJbR"
    "jN3tM%M}iV(kQ}h;a&C{MN&xnW%`#;A0^-x5+w4}=&o;~S5qJxuG0#ESA?%(6ib%HiX*VDYW|k1w2g=wo;&-Q`*uhneN#`9BnIl-1~-4iA{w%c"
    "1=zxVm^na#+eO`hxNa%5^H!a)*C`)dHb)_Yi^5w?M&IkA6>jJ<MxDE&(?vQl*1G4n<}%|b<C2w#eFTSfW-~NHv6YYQ#qr-T0BZkzbAuZUDFHaa"
    "ih;sOyht5}kksp44gg38y|>*OVKqPSbxlx6&8C>U5n7UojwEIV9D#0pAn_D68~3JJ@pOCVqEIpIs=eDE)>Y}10=Ib<+;^{I=B`hm6uVd3fH}G7"
    "#OiTkb*%Wr<HG9D;gKMTtsWCr2X#fKt`DbM1JNM=W}8dn+QfHCp*W{JT$HPyvRK>=VQ6R`=8F8WzdW2wR11Zumi#1FmL8r5Qe}!}wCJK%(9HEz"
    "h{H&&z0xFjTP%;wk8XsG#3UagfwI|bz;_z0+b!5m+oo8&nrGKE<LMxp@v7ov^=mjn$dF5xU0aR#{RF@h5zB-sD@KrM0;rVbB~j7XYa7NJ3<GN_"
    ";l`C5L;{N^P=w{^;k(2{d0aj(BNgS!)JmxeQ;C)_QOP>h;3mZJ0n#%cSB-vC2DhY%MW4(?lE=+QsE))jS^yV`j&5djgpWfAXbqLIOTkvu!02v@"
    "BH<^uJZc55{Ofz}g2ft*-dLrM49$sBZ!WZ<CxB_!jGP>OyCpncdJn#Ep8NtpZ5xg%-k-v9^Q5p`q^h0z$M<V?-d)pfaeu4R6u4v7Gn}8aRjqak"
    "*<we}nAB~+32SNpbeUb4clm_i;{XgNo+-sLeexV}RJw{8i9AeAKt;KiR_nwMfojfR2Nc=mvhrlnCUF>mL*iXo*9Yo+`P%oGsl|3*vxLYBULreU"
    "mI9TkBobA`rMU&&r1V`NgwB*t4a*byB=MeUx=~!_5(>OvErz#Hi1A^LT8TWV8GC_F_UvO_V;jMUG8Wh2IxgONAikvNJ_Z^-<dL-<@SBQ;mROUK"
    "Omkz-lp&ZiqSb{*2f5;R0jbbw2>@MG^%_kI&_uHrLfk$ES+Ouc1WG9TZ>V)1zm<**Gg@&iuu=^SD^2nX97(QA+81I*cW>|Wg;bPB?5tN*H+kEX"
    "mrv&APqN5Hs@3MvQ>`d0y@c;?H`gjCP15glK0vl2R$5E3dcxrji>__U&Sv4gWmOA_49iSL!mrh(YGC1-A@75NiJV+dE<^JvTFz~<axX$g+K1^}"
    "iYJ?F72qnVNY*@;wHupS%2DBS%l?)MP+3JLYjxtLe!#6vDz)Yti$|G?$Dg3s_G0WSyO8wKeOfX@%11Qy%ARau?LNMTWd_CIEUhyfIR<~wGU(ox"
    "aWC<kgmeJ@WY<Mf)1ovmN4MGxsRAKfWw}t1hgIO<B?$vTCYkOb;kPI>sNg!35wV9Pk3v7-L`^;6&{obHaIhyL#c#L9usT6AohPU_D4>w{)w%=_"
    "MgzGK)Wb6!0jwi#w`-S#TVS=9(70OH`|U*@R1JCKSIe0GZZ6y9m>MtKcPTl{SJnl_>Q>Ks!MaPTO16jYG5ajN5*CN|W&{811X{#XB7U1ws73G3"
    "2rUlIUYuY;+aN%_b{++MhVtzf#kto&KcJ|SL~HG4Xd(lxXz<)vLO>`-oLDUq)lKK~%n*wnPb<Pv(DZ%PxgwBrofKk09nE-%-vutgpMqN1NEr*#"
    "Y?_~QTj>ZE?O9-U6^-CM%1`RsgOI^=E6f{-Ar|N2u0wsmHnt9G45leo>{H|=i{j@5$3?BQavm@zi5j_b;P)-z2Y#cFlo$>W5SYXjwbC_7j!Bxm"
    "NW$w0BUomtXl278y_jTTw^d-ij43`qcM=GwKRuTTa$mai;*DmapXkV?J}rQ=1#kjR<O}|vw4Ivf1*|ext0pWrx<Y;Qt!ZvpXFTc^G7UI#c-x%;"
    "+~rk5$9&v}%vTcj$qdPX#MIi@vAYEtVQ64k$G!u&HCVgJOOiB~u6+)BX?AOl(=A<vU|5rR@ubPTS54-Fu0F7cC{wqCc}%)Y-C%#j7ylk)nbfYm"
    "X6;k04v9xr;CzV$7#MnV*rlbbUw~I6nHspY?x$QN1F))?A_y$hdG0S(N@#tWY&(dAN?OO{f@2~zyq0v`Z;%=nsz}-qpTg-xyMjcK>aECq^vf1r"
    "9P@@tA)^}QC5$Q=DT<b+<*tLG5VeO-iKNvInUVklx`RYmZeh6gTcioRQC#BrML{Qhn62d-$=Q}(T^it^xz!4>Ha55i6eqPP>Fk_QZj~5t;uo2L"
    "lqT@zI`%Tl=`JNu+k*I{U6Q0&uObDN{}de^ZJWl&BKNrVxv{B&Wnr~68j`R4^jtyKn0FYolcNi2`U}fGbKzyRGU<Kr#a=2^ES{vC4-;70pB!Vv"
    "t^k4(D>W_;GC}C*2pr6d#3&bh5uGN5H}l24!WdIva5aD`93n#X?PRSQRf5%htHo(bK#M6&XrLPAB+3p@Pc}+pKp(*yU?8>-q{nb0o^etScV|ZG"
    "bRlug5%wi@3K0u-E!BPj>IskVD7Pje%$w~JHh$<BihRphDjjY+cGpRIL9M$4EmOv?nW$5WB3n|8R;6RGYH)?Nfrb!?72zUL0@)Dipjezmg#ZbF"
    "QmF-+%GCt)ZAz-5Vy4__DOrG06b?l>s=yWn+z1WXpq^-h*TNZDEc3FW<(j}By4YfghfjzouI*If?WNZGFq4;a1mu+uw}KIYNU@19uX5WoyDo4a"
    "Hd`1eJZ@GzQ8f}H+R_}X#Z`tULU-2BBu>#Fy3=@tdU+zw&7(kf20<P8Lu|C^#Q;@wXS|zHketZ5b$fH>sWjdt8&BHnb2T#4O+>|AQuI4g5@^_`"
    "K3A)C@m!!`u4M?|{xI$jb-h?d9jTUMr`QP0NNQso&9V?>aZ7j->Ysw`LJ?zTy5W6DC8&fKqn2pQ?Eg-mWqJKp4*USS04YTVNdcInsvWVp&*)GO"
    "(pPMQG8Rk~^`5XcXrU%a$wINu8fsB0-h_x+$E*(@t7bfbDRbUBZ>c6MiI)5_%nA`%%uCW0tqfO~srA5IZA|yjMj|H-xCOuC%~cjHGX*^W<05{D"
    "UQl5!bBZKcRTl<JccCs*tWut3?8I@nwSfY*S3V)rSutQ$NaI;6MrZm!UDjBgJ#<7{N|9QVbDTsoKdNKm%yR>+c%}7#vDk*HrMx^EQuTO%j3SBb"
    "V?j8iJXA!cY1@<mAav${)kW#`r3*JNXu=_>WYm{w5sUuRlFgK=sHZD%s^aJ}sLmW-_2uyEy_|Nd86((RXnS^3L&Y5`95+q5fbg9>?Yj;@^|3pr"
    "U{uNOufoE;taS<uVO=Fbb6j^oEJ${V%Va{q%U~0~JnfUfFbp?!q#5=n52F?KC({S6va1<aTB>icZE3TZV`oaep=5HDhR*&^>lhLOHTjA6S7Fm$"
    "4eiX{548>P)P|=voT?4q{vTXbWZV"
)
_V10_SCHEDULE = json.loads(
    zlib.decompress(base64.b85decode(_V10_SCHEDULE_B85)).decode()
)

# Current radiant-allomancer trajectory from episode 89474114.  Unlike the
# v10 family it front-loads a cheaper 3/1 livestock opening and proved more
# robust from player 0 in both the current train and chronological holdout.
_V11_RADIANT_SCHEDULE_B85 = (
    "c-rk<%WfP=lKdB*dFZO5q~+e&QcX)3wkVL)6lNMiqk)~p0*l#0&)gRK@2hTBWo2fVo11%h@}aI<q4>ynWrVwhnfXuud-k8d{Q8f-|9bZCKb?KLy1P63a(?z-"
    "zx>aC{PXJ<UqAlimtX(;@BjMx`KPn@A8xl_e>(c`{pY{_eD(3ek5|`c=Vxzkc4z0S!`FA)?YpmE{<yuq`FM8zdiM41{r2kqx39na-}$%!-+%sa{qE<l&A9*k"
    "`y+;y{B*Y4-rfKH(2qB__wUcXOxyO`|NeA)^X~K8`|-bzHFp1JujZ}#^x^HFKYtqi)u>tf)|`(&Jv3El;M!`}yaCr&Z@2rOPM$s=FRvAM+w1G&<A?qW4ZC~4"
    "-F`w%JB-coFGuaWyZO9tP21AhH@Svxj#E5q*q?rz(@A5d)3~0F*7j<@yn2Ss9-U$M#tnLV)lAy{4Lp3v&fcsUhyM@P?bpTi@qTz!QFB-;*7{(8R!kT6*TKKK"
    "-`<AXAgtr*c4WR_57UGvY1n{{i@J1p+66m$5Sb&mAEuYezPsqP8FvXy)V{g7;b<DNA8xp0hp8AopqbE~a~Rr(pHCjRpAj_t=7jLx&-=I^ML6fjAs&5FPnL~8"
    "ob4daKCrHA0>e0e$(7NZ{kv3;+rb?s6prn~r%<okQEj&ida}T-Tm%O9_y#m%I6RmR=Cp?w!YCe>$uM^F1{$NO^%k5HVDIHGh0`m(T4!BpziBsSeVs}p43=lp"
    "^H_&Tm?ky)yovwa<BLb^;9%en;F0F=-o3rKzTUpQ|M^eb+xrjKAO1F3B%$<S)7vI+VA<({dXgtc#Tg#0HObz)&m3AeY*`?dh;<x4a<&(%p?4&les_EGDe;J5"
    "qDL)m0|xMt9rxhFKS%pBdvH2>pG99JFzb4<)M@hs3eD_cljWV9ojf(YM2~KJFnUiD-6KI@vUi!O>g&V)6~<{&+X?W){2a~kG1!CS6QyUnk2r>LOFe;hA~Q!y"
    "CyxKO{n_*EL;pI<JYpD~l3{B9N!S0Eqsn=BdTyL|<zc2A&Wb?Mf<dT`96g7)R76+4sqmCyT@~>um=6W(avr_%a332DNruA&m7puFE5NV;GY2jR^hf#du#r|r"
    "x&#S`O&5!CkNHtMf*kfP2y%yqqX2PsI)#GVW6MGRz!gwp6{1J9OHY478|4mvdrUCf|4SZVAGq1r7sPbdT6v~HBMYr-V#Z;zvcWr?HeqsR|8jw>;VGW*oPyQW"
    "69%iNsPq(-f~eFDZTdh{7L>*VVk0UA#~b>Df-D6=m0--}VpImRRfXI(7#qJXkA<l8mhKyfP$`5N=pYMG!3wDkQQ1vFpcf=YWdR$FQ5TIx_0dblqO}h=#iFNJ"
    ")C@d9jCzViPqC=S3|u@IEsKN${Zt5Soe=5!tJ}YLp8+o?gDj$*%jcF)?=3c%*}m~N1*>xdbxvboxdD$&vqDhu@bu$XL*vY`xVyi-+I_#hz5Q!T1lC!nFf`gX"
    "b@T)y&IZ3|w%3eyY_b}92imwewn!P{Z~UJ|3t@X_G3A`)nv24Xr_6gA=rSS4lE=2+fbofqG~*hgl^D5^YU6c^y@O1vO}Yc*4D<iBA9|nA*9mqlU(*Q^{kQZn"
    "5YlGpW3jZx{HiRkk-Am{<8D^ZmvrFCzaA~5mV{Pj>vI~h<zns4La}yx&et<{1F?52txJqmcF_&5x6w?d0EDeeWNXK*axT<Fz@ett>}<eM@8@tV16!+>vY3c<"
    "63vdWA^N(=(-B)NSlz+nqGb}zqAt!k@8=GwTkaSLqCw1F)v-2)9_x=L>dGK(HZ+<;rZ~)8h_=~#D?wwMiu(z6|H4|4>N3a}$Tc14@uTulXiHq!wu(mkgb!MX"
    "7~DeN%abl%KXI`Bx)(u)s%Z%&h1mWwFn1Tm8p(XJp|2B&gVVVeb_PgeAQbnX&SdT5U<Y)KKfUHK0o~s<O`=lNNOT2IHaCsvxN0Fg11B-)NFoaoY~<Y2-fOeT"
    "uxoUc;HJ55Z^?kt&Jszyj?4n0twj<^jV6<2Q<BmQ!oz#}&CNAC<sxrUn<JR0UONyjcmp2IKtq&H=yphf%@V|Tb}ADBW|#?88)6Efe`fhnnIl3ODB(e{W>H=M"
    "hUJ<+D`H`Vlxvy(_C7N&`)KF7rRua71oXjRQ`xk%k=2)l!}V29?D~uYd5-f5LH(-*gZemYIWexUx|ua75V0uD{SfUD7jmgA9iV^U{Gvo|b3}@xe6O#`pZj;w"
    ")ItM9<&bboZrbxH0lkh2QK}=H27b3MQN%ma1UG<As@Np5IOREa(UEss6M*KdiUNu<yEXxdGzwxB@HFyxn=}$uD^{#E#xpp!YZmTg$g#;pInR>Lj*Sw+i+Ha)"
    "223AN@bq+;&mI(FTz9y(W}Btp;~mW+dW#p*s5W<rJ7S$z*zsm)_aJ|I{Vda)eVeYYKEA8Sbs(p*JjL-zMdv2DcO|l&#k^@)r3vPZX3P8IhwDGH)$^t|S7mAG"
    "Y)TdSmAhA@r_La%r^!<A3Wafy>?@=$yInCE_1IIxQcR?$?P8#7u{(>GvZOY#4ZvnKd_$mQ(2<h%2G{u8S;|qb3<7ZQ#Gso>Son=k+u@q{wv7x_VS()k6P6Z$"
    "x&TdfI4lv|EN&xx_<dU6Kxk+iSs^l7Bbs41xB<XV-H%~m10dXiH$+rFar_H#TuGuIVMrCGu?NbS9r8wX?YFNTB9ShQkHq<OFgVpNiGW6fo}>02X?8BVFd3j{"
    "$hNqcA3#wvQmQrOZe;~zwQ6gdKtrHF(7Nf>1Lw$A&LK>MDIn-Ap_nly6I{n~tPjE-@}x2rB5ex#WHW(tRKD<Y!C%(R4DF}Zn83V5GyPFYHdC2jux!=ceKd-i"
    "6;pi|$&_JDtEmXKf*r`iO;LwFALTbu;ZdD_YIHu(t6unJ0}z4#Mt&!=C@y>Ng*i3VqFP2Ru;MsE<*cDb9FuKw-&{|!<o#$73D_SH)idy^@D4jX4XC;r2AuTa"
    "ueb*HP=Wmer{e>_0^>K`7lcw+b8@p|IM2)@rVTbr-u2NX^Xjw#<3iCadwU7*>49E(k-r&O88KZu;%&Ryw`@8C^yMTB%^z-V?zX|Kww+@Sas){Um^2x++oS|c"
    "ZWXO&P)%RB9AiMI3q5SZZ*S(Z{Uc%T!w-+)4~^a#^hQmGt&Oc`QM*-Ete6cj*LG425P~+MM<j=q6%dKB2f%SL>@36UYqVQPfs&Sj$H=ci7ltQT%fdCineR2r"
    "DHoXK26ucvQuBX%7kiDp5f{Kqd4+$IbZ(iyIgRS}VcqGVhXtr;UPgV)c5pGqxKhlGC4-!RSEfC8$V<eQNCz7}%gSDDo)Ef-Jp1|J5UW*=weWa|1I}_O{2rLK"
    "x=wPqLAZ>J8y$M{?_-%g(?$(OhS6{`;hpufypQ6Qge!OfZ(ufrxur=+!30HIUnaxPk{^<ICY#xgew$x<z#KTUVpAAiU#md<7NW03sl8kkDY$M3OOYoN^b%d_"
    "lMnW==#JBC1c!Xf`hy1Zxtv+Hpd`MO9{A%okcqqrpkcvfoT5J}`+CUoGm3a}>&rv~DA6ZZ!re{<Ot!A-_C3)7UK3qXbdZ!&&m~qQnDi$(lzhSf0k9=e^JJN6"
    "VGm^v$F%B)o6wD+=nPs26a0}mP7WV10V|PI{hM}Ox*w2>kO03)k&j>*kWx(8BgD7fu2rViE^qs_j>tfOuOO58o8p)_{Go$!np6W(wOXq{tpnh?Vm9sUO{2a>"
    "1?m*Qw-fAEy`MG9+Po;jq63AZf)gXD$!trBz%E712{!-hi)N<<oRgJRbL`C%0y-Bj6VO?|BtVBbnRX&O>qkO%GKDI%bbU!WN#~N3jmntAL26oK-8}A;8W@DI"
    "k_5Ow5}JNoi=FavtJ;5}DLodb_=1}(K}Gmap(d)&ajjb{BzY_7d?wpDpkBO%$RO0G!oJ()VgegvUS=vRo{%7s<2YJ9pJ;S#&EnlQZrv&vy$uf~Q9L^1-1@`A"
    "Mhzfuj(Jp*AT-XATrh{)Ur_ItWjG!zE#{Sw2#Q31Yxu7;nJk?ij?H=81`5+s0fPux1Sg2<E<}9XU$Ew7N7X#h(orLv86my^nqamr{HP3q6oI*eJ#$=<=o+IH"
    "GpVb-xjRZku+|_^e|^G^GjV^%{33Mki2_$j;^dfirnx?OYfNp7X82jvjdf^=xkFEz#kf=8*B|y4&3;Lqq#iq#%Q0;BJKcL_0cnnUD0kkRz_gbKrg;jX=M1L_"
    "tQs1UMdyC`MF;}?wt5Rg&g$CFngY!rvM!?dmV=SKIt&NH@;2Wsh~Q-1$~ndL+(XCcQu5494R4z92UvAB;Q9)H3z<Pt!I?zX1s0(@7*csiM9fZ=0wqmgQx?FZ"
    "XXvjH?FPyDBfVG)_ra8FgSH(aYqJVMNAO|LL{DwNlR(gNs|+jH9g<9)cJHI33!rv5s4kS1i%+Qs59j5%yNH|IZaZ{Kj!&(ODN;I>ExS_L8=NLp3y5qZ&MyM7"
    "64Xl+5WDUzl>&1qn;ulMu;O)MYPy-4Oef^Biq5_SJIRIuygMDzi6SB#hp6;MR}r!pNSHf-8GhE|LL?T+l&rkMU|0ZB65%`_RP3qn5Ht1JTw43~rh%1`5YpxZ"
    "A${u*QXg~YWf8KL6xQ{jM??RZcB*~TmX@F~B*{l%fF(wd0#hT+Cj2cekHml%q$J7Mmv92dSyW;*M1|uQ!e;XI9u=h%s@W+TYOp$(X<uws7DcW~<T94p?wk8q"
    "@xTXMpimg-7_$R4kZ3a>tj!jfHSRk@OmfZ065ieSAPR<Kmr&`|&ACaWUj^j&#bTWz4Nh0*U&Yci2DlqD?QXD?(IStyQyggf!w+D8qKPPfi|Np^l)q@@FwR#z"
    "xu;EUNk~;H=QaVI8&7#@=AmsKJSY<yGBu9{V;U~CGQk;AZ%#)nXmv2Ha&u#EM%Ad48?*rXO6OteWYxCJa-b}q(j%%drE3cj{}@K<BCM=L)DaY66v1wG6ZL_n"
    "Ypn&hZ45lgF3TD+y;?-qM@a>b3ng*DB<@o51mKa54n=Sa2a+P@5|4`|sq<!C3+pgNSmM{;56y51a9mUw87m#t-+ab$5lgN>;3dRvp_O}+&<MZcPgG}$7$u}~"
    "U51N-IdP1@vd&D2%Fax@#_SK>>5%c%JQzt`uEE?;1ra_pXY*QPa7;KVfFu&JPjj4byYhLe6JqP%;~^v@n(N3-cMC)?6%0vjm_1R>A`6rGIKh&#KQs^_ux-e0"
    "bIi3HEFE`0a_^Afe+Z`zl*C1mp1^e%qmO`nmt}OKT2EpzJ)~|n9;{b=62*TBSOY9=g485<S^h?18!u(DQ5WqN0iA<YSEK`!0!K4u5#S^^K|2pqEI{MUWqKX8"
    "K4Jj7)MvkK6vu9XB?Mt?Rh(+(50}~r&V4%-j}1-$Z_;7DBUnapm7Jl6p}Ev|l<yq>re@g_F46dIVtpbIy?g?Z^N-98=Oqz{^!(hhDYcoqm6Ipw&m?8aJYVkj"
    "GD?t!Kn6MUqLuwxi)3dqY!<`<eOgh6)~pS7b=gn5YD8xsO_7#!y_FJ4iSdWt)haJ6Ruq>wUC}Ez_|jXI&HE~I9G9V^z6x@}BF^IH<OaItP2dJE+pOCPa9HbD"
    "@4%MFd%bXZETyDg5-3*z23lZStOK2VLpOn<T_Pamod*CfhujobcgwU#ii*@*8_J&+*F_U$u<Cabmovgh6QwK+lBdjulypcmw&LFJbVRB7dx?>Ang~TlIA>qa"
    "1WpWE7uhRq2gTa%*(QT&96*zn+6G9s^nhz>%t6@T;)`CQ1kD<Nh?osF1#Gp-Zp+wWH`og27rAK59y&SMWIsbxg}V=zfhy02OOtW-QO!!|HvcrnKDt>nt4hqu"
    "+?3h;gzc|ELJ#K;9CU6{0}=(JN?eXz6Uav!R_{WL8b%+0QU;LSf#ntx95@C6nIg=DdJumVc*N4-txQ4<@cJ<`cvkYu4oNep1RHgoa|qX#F1944UpuCQaaqI?"
    "RFs7y7!VipvxuFL<n+Xdl^J>qEA)tfDP1V-N);2en?ei3<V&~}>vU<Jv+1;I!)_U+VOZw-rnr!MfN-Rxy~-7~IQ$j0qM36!IHy1wX=qTh+et*^0?<w{tbs7c"
    "DL<I!A5!TGwOrx|BLmYT6h9Fa<4#IqdC_eGwR1(fny^Pi^>SmllWuk>p@_nXG=-}-!!d1c6bxpJ`dlP@mV?BWXQXIylK60u@+U`OR9oaOK1&K9Uq69{YZ6}^"
    "eqmbh6dK8hfjM_&=_*+kS&td(9~Pj4m<3*m)>JJcm5?w<GoDn2+<>nrm{M8tJv&pxkw@`(sb&O-1<HL>9Wo~^Kw5LlDgk~xgbbNB^FIq^v}O?(NxDIaK!h~b"
    "0Rlp%46<Xwjy?d6LRP^pfRI@@Ek{YXP#@>vf))=N)R3bkm4P>tqDyL!hp<@-$)!(#9cwfMvFwq_w#a2kG7T%DNV58M)5QSxMNUls2yQ0TiD<ext%;jti@AKw"
    "hR<?VYQCqkfdU$_pwW{EYKd*nz%+m?L9<#3d!g>!H83Nkg)vx$LJB<CpF<VNEsc5>i~BFEZZ|_JvZOldOAOWvC9&E-Pu!%5!e8R30=TJlyf!&E^^&BOu5!GD"
    "gqR3qb0W!n<KD$Dgd`(CGD4)nH|x{^xhQRhaRC4lkRhFPEkfd+8z;)h?2L4Es8CfdUCpL(nPtF2;mv3T)Ty4$G3W4xB8O6tb%Gp<wZhp6TFEM!0lTOQaB}g5"
    "f_U*%l&+_`c-3eb6++LDBPjVHU=Ir5IghEs0ZPR`r-l+E0;TjTvE8+X){Z$fcOZ^>F{LVqO>q+Lq2rZZWX-HF)+bvdws2<3s`2)Y@64^=&_Fi!DpJ@?A}p$C"
    "O7SpLJfG1~MU_}ak}{nf^dKhs3k$;Zfz8&T*7?OzIbMWYLMcXUssXM=J?&K$py-H`a>oU221dm(ha1udiw81I&mw3H1G}iRl0-s&l$ufvTL#!tS>{JU0?l}w"
    "U!NRf{=kkgPt34+EX$ZsWc*w$W6bKX%R0tDVOfTg(d{ZA+|6aN8!@j6Ev*`oOq*|0Jz12fROT7XYd-5f4lJH#4yYNNL-LJDJKyk>x9T16uqVoXp~x7$8q^2^"
    "J=zpmPen_OMAF=f99=7JB>Gl(6__pIYqmzNxEyFvdsgnHi`@w_8AgDi90qsq@zl66Oq*~>n9DF<qqvD(ddXOM9LAwTNP}kW5yA76aiv3p7Q8d{6`KTG${dGb"
    "j7^TE@OuG=L$28agF_wysxU&JHpv}Jn8=h7XJ{KMxka~cpMbPs1(2LM0gDKZHj@-GgK&EM6L6=nI;6&@q4F~d)H=k>4q*kE^VOPTJX|&ft!p+pXu>qdsZ>@q"
    "1R~td%}HwY6jH0sE5H|^wX%Q3{s~QU9$Sp1l=#V9Ukk>cPaw!*;xIz{@Hhw{Wm#I91(K6MsWZcE#0mt96a~H|1@jb@e1KbZO=O9!19~jRWYRp7g=9R-J5w&1"
    "HMiajqB(jNyvr{beQr$VRGOwdkA&^ay1)3kZV2pIUr73<YKa*uV&n?P*o8aP+ARTOYDfu1!bMBzKDkA(TolaNDDF@zA_FMSHij`cMV~`e&nW9+XzGx*THF_k"
    "{`lq7SS5I#o~<LsMaQ<&b);dd+=n<rH39$wElLUCALjJV5wC5yI2ygKGvtsAPGe_o%0=8cADr{&=q=n}RmfXdVBxeJlrS2x&R2m0lp`l7n3ZdCjfaFGIP}h+"
    "JudfB8cQvRX0yx6)J^cCc6u;}r@&IVsWcAO!)pLey*??#{=f>cm%%j3fJ6fJ^vL|9$QoVF%cqQ;<g8Fd7Q{&~8Lm*eH~7@os(kNM4~0HBH^N3@;ELk<z$;o="
    "jHN|MVx&?O4|jbAK0xz|6gF1TdRWTM<&!Xi^m4_zX2H2b`WLNMq(a;a!WDiNg3ve#vXCs5Lf$N_u9mHk($FQ>ou|iwgHUSqs%TCIVlE&G@51Dssu8tX?WpvP"
    "z&6fT4zTt)8WzA2Wq?@c+R)TGG<Q2}lD3pqW_e@^xqDP3W|nx)C2EPw2+DAQ_Wt9gp@CAKj6QR?FrpHz$WB{IOZ+9OC5HG_?r5vFQr=716`Hg(E8VzG)qYYg"
    "54SzH0j{8)uI*<llyh!uMFfdm4Ae`}E_LDAq*7K3Ky_%*E>2psmqI(L!iQc60jUHRf<IZA_u_6SyF_s-&7p?EMBt6gl$H%O8+pJ|B7(yg63s~6$cmXBJNH0("
    "N&|ewHKkWgC0`~?13%MsUvaDC&U!upfe5=T-#T^aYNTpBSA|IeC@e;ShKP+*nO9QNSgq`@V9KS~`BYa{Vx_kofg}U2AakL(YP4K&o8|S~a5&_23lSCm7Etk{"
    "4Qz2=q7ME|P)`-F?`}XeJV%{#md{I27qG|h{y1*mVEGs;U+vFYvX|FQ!dg2Op4ln|;}ae*AQ@YC&8cE^Ch=}OQnS|k_`EW`opmpV+f8+iRH7^NC@@P5Fmh2j"
    "O(Y4n@cFuS&3zCbq-)whRYE4&vozmS%0B=)CMG?>J~p~PPhf%x&cwG86>z8eIZA*G&m-xJF;gnJ)Vt7(bSN-6VT^48pS9>CdQB{9-7DzIZA?TD^Qyag+WI8v"
    "TSU_LLM*wLtDfeeVIp89#TbPp#%!e8O*t5&tST{2bd4cDUed@7qS~+eij8QK;Htu8m0_yb*d_uj$OEm{-C=S8hryK5nK1L}mV$*KD@~4<xkqk87Dg^~?&fj4"
    "!c(Y9X<N*g9$eIu5&>S1O693Ecr{Iml`Cjfg;h-7lS<sPWyTAT=>@el+bVir&0bK3ZWhab3HpxLs)NG9^Vk)O)q$@Z;6zE0OS~}FT2}5hPDUd2G9!vS<+_FZ"
    "K3VyC^+NR%rH#==x!V?AL2!tbnm{v|I8*hBU}d|eqGh;Z){t^lb<i=U$C6RlskKv4_oye5x@$cAY5dwHRwfgv^Wf17;Jvnm2GSV7vB6%6Xtls<iU|F7)3w&Q"
    "o6;rIhd*6%{E5DtVWQWGzx7AqZ)J)kf6Mf(0GVU~2l6y~i`Iy%sti|6*HZaq9fmy}Os)vj<Uj=lOrj)#%Y$EOzvW2{5xD3AjpaT@0eDA>CW9M_^SEu0C33lR"
    "bzi60#!H#zNDK!e=_Ryp`7IUaDtGc`0t8wVhao5xs2PKdncz^!OU>;G-m;T78PaT+w++}^0E!@`B0Xl<LTv`ZDQmM#{A2G3=%U8N>YCH@HD3Zvb4ZT!3@p*5"
    "%VYEfu4+wmTbx`(J>WWCwRRqa$}IMityg*kdKN>DHPa@Fc}Wby!1QniAwLkni~*A*;ZUjJ3<f5UVJ&QbreKx{#>vLr4il}5jit>gT6+wNL>#ZmH?8f5eU5{{"
    "w)O?hls)9O{%CA0NBF1!WIn31j%y)b)j?;|3!3vs;A@?*v~QTYRqK%z*ETK#ov~zt8UY-9BY+C8ykH-}%>oW6G8ZIKx-X?BaJgbq2{wU8mVs)>*{>NzL0_y0"
    "x+xdwD<Gn@eGA!o31Y~c7nTxp#_I!!0lDjnq%W_FSV9+&&9-2K9os1|MKR70FV#sggenrStMQ0uvv&S>I4U@DAi8L)MqrD94IVqhQ>Snm^whj=vALmDp?qV%"
    "C>_a{UxpmS8Pn7HEJ6uzotgG}Mf1~Mi%{2p&iM&fr0oQQ*3o+sPH>}ef@)dkRA)_hjVgc1-2kOGqJ7>W=GOMg1CY5HBNf7mWhiU=!yFTL%8T%vI>Y$jvWL8M"
    "TQ?wIGb}PhZ`?!x$I#4SH#$eJ%=+7U)mQI?T@c--C9FB1ou!v71L`o>`J|!zMl_V)Y*oZAq|OPW9$i@J!6tE;<*uXT_kuD@ctdM?-nTeph?Q8zK+G}vR9s0A"
    "tkvlLGCEPP-k_S-6OI-wxwcaTK9*Ki&E$p1ez7{4T^@=w-j}Ia7hh7f&5(hT;u5H@$)!pVh$b>3^+8%SwGM5xv0Jy4nuDVhzqxJoFNPe5HpCif$q}&xc)L5#"
    "dC76S^Lcee%bsxmykyx!IH>XE0Wn@iWJ-(9xLYS`?qMy`PSa|10Ve^MfKNGabQO|uk0>u4TW>3CXPhaqz?~S3J|z6O?M%i^eb9z$4trdQtdgqwSc4TFCoj&U"
    "mS}WUlWj92{#GiB6Ks8l@lCW)6g7)KWVpTGxH3l9!&x;k(~@ccbePz$zDoc_Cl%x~XKp!o(o3aoX&jUD9%W&#Bfa)KnL;Gdv{Dtofo3jKWHC^mP>^NjGYviJ"
    "GYBW(OV4$s*R6gXNLLfeMhCBvowGYiGHUU#ezVU5jE)>UQo4l@{XBew$4a6lOb3h*AeKo`h-c(#qz^^>3#xfkRD<oN>bd;PGl?Q@jHh%6yA<xUicVY6)<>Np"
    "fsV=$L$`RC<7!3uLaav{y+^tO6Dm`QCy=C|4r@rLtdQ<0n>+DRjUi!Mr9x`y)B&1;g=tXk3_ZD=!?t>oqNOA`Erw_HCM#pP`N_D4i~W}hrFQg&LT!AY)t?H*"
    "95PbFu02Am>EZ1F44YAiP9>{PqB#QPp>;nn{A$bDQk9{X90;=1evxsYG-?_Iwu~%KW!ZkH-BP-+x%HY-xyJCsB|zg=y7&F2&y-9|+>_#n$AY)I5xP}LZGkF|"
    "Q;*KFLB7lkO-T$_PleaTtemCDP;+qW?-WB210ijUdqOJ;6&m2TA3O}mp{uBxr;uZrh}h~2E8uaLD3iSjNUM~Y;N`LX1WR$gSEQ9->k$?SN4o65JY`Sjq~r;~"
    "E}I0`W4po^pVo)q0rSM7c8)A~Dbr%aD{7>z0g$5qqtAjsYFy|3fV#l!B$1%l)&k|G!+rLlB<=ppA82I@w2~#7ygMuinFsAI+rtZ_QF%wI9QNYXI=yihr=}q>"
    "p9rPU@e;0#3=(a}oqQNLJWYpoiMlY9ECrZl%=F7Q^-Ne}ZBnW_<|oby-!!u{!=c<1`@#p5TIwkCYzH1>S`|jox%~~rU#KM#n~7ul=lhL`kT3(T_8-JJfo;10"
    "9@n;vIfH?T5r_N&;wVZjhWTRh*;v+OuHGbZH&3DmlW@4BD~{OH4n`lV)iBcVTjyj;^xtP@c$ZSNAWh;hhyoS7{0g~Ki4Ko?2NeY9v%F$0&iCR|0(|8fqt;5w"
    "T$#^WwBP)kdN=H#67L)-dJfp7n8wcNw(LBDHEhrkpxuV<H+#2nAPiu>obZ+ce(BC@vMunzb{$|14{VyE9Ms^U;4858K2b|TM9roh?X_DQH%8mxnk{fQPAW!f"
    "dq6JENEFt76EPwzYJ}xe!N+hCoh>y!<afk8rfBNO0=1eiWGp33ON?y!qDl;*@7M8o!!!HvB*R17@a6vjNBAKS"
)
_V11_RADIANT_SCHEDULE = json.loads(
    zlib.decompress(base64.b85decode(_V11_RADIANT_SCHEDULE_B85)).decode()
)

# Higher-alpha radiant branch from episode 89475210.  It shares the first 109
# actions with the robust branch but performed substantially better against
# the current radiant/Pandey/venks leader set.
_V11_RADIANT_ALPHA_SCHEDULE_B85 = (
    "c-rk<O>Z38k^C<_^T6&VHPYTVQri;D5e15J!yX8O0qn&BhJBdb+hYIwYQ$!BRb^ykWWHBqljg0_61(1aWyXt${P}-R{`1%0{`vRcPX6QPlTVkQKc9R(J^8O+"
    "|Mj>3{`SSUkN^Djw}1Tof4+VG`Q-hFo9(xM9ew!m%U^!E{P^Lg%d3;qlegEqlhf7x*PplBci+DJX?u13@#OT??Cah8?d9$7Uw{3-({TfS{PN-I-7nvoar@=_"
    "BZijze6rhqzWx5tkJmT1?@zu?+xGi!f4aGT_vP*F_~&zt-T&Feyj7n*y#3RcPouvYHEZ9R)A6UfrV0&Qd+nMx;Og@2cK@%Fr_aYLYQ^X6)z#tgL;r<_eSW{)"
    "enL&VADhEpj@tM6`pdpGwxzRgY7N~Srg+w{Km0zYlg3P^aXlTa?d5)XwG5j-I>Ydd8}#<lOxk`2-hZgh-mDq-zYo{#x5f4GcG#+@Ijj|HeK0^PrVIPq;9uTu"
    "Z_;fL*75W>ayQuBG~r1a4xq!L?m9en!Hyn8<_MmL<)yOkF4{KZE}@CqH@7w%rXl;`hD&ysis1vA3GF%eLwon{lLzi+1P#ACA-wnVKJG^m&iP@8N1xO;%O)O9"
    "JBYIntSg(uFwQr*GMclWEA`k8?l7TnZ0|m$dTmG59vAe@0=sga7~I1SFk?78m=5N&hv&j59+t^4cJl^0qp9^4yeGif%Qr>PE52H1U9sQTgIQmv(g=g)X?mXP"
    "FbUJ7MxUDaKYV@hh#kBb_yc&PIqbW)*H>5Dx3|CiVS97?;p)R*CyOMNK5Sas1YTHvx}d(vi=z?@kJg%E@55&fts9Ok5KH7bjvqNYi`CFO6HdRox&Bmm#4ypL"
    "7TbUUY_h`{eE;L<d}a?WNAI)fiv(s(PnJ6EzJXFRdw<CCPR`yueR_!=ZF?}<r-km3Ah6iG%2c)WVZViOnq)fx{;@tsb9@Z;;P6E0+2JD&Vcb$rpq;49k<y97"
    "@7jMm&OWrSv&tie(djZw_D7n2Uyds0;pwUKzUv-l%HgaC6fGEp>d4VC#HAv-@~pyBigopfPr-aBSXc1qRfPNaXh<;}Ca45mX-xr!510jTL83p(hr5Hccce>@"
    "fZTMk826AL*%9QhcZVQ%csL3WXK$xekb7u3=pVQODy%~E2)p$3C)g-=_}fE*X@8eIz<=RZV_y){d28iJgGLrw_lX&Y$*KnLaN2~)nf>GfS;J#I;h2Ke)guP0"
    "FHz}BR0^WfFtlj{O<7Qy4~U(p6ujQhClq8U2&x2QE*GORn5|dH?SrxT>+)QPN^j|&L4-;n%tQxShzeFn?+}&U6eM~<VpJBe(HM2nSX3XqVl3MC0WY!WODyUJ"
    "o*+hjiA7&x(TEwicre;65>E6}A+R+<r0*|p{v3S<yqpTMh)%AUTRy$F*kES+CO#Cr&JEN#je!*gJU-0|LB+$<4_^(9bI0QI?ak%x$L-C{Un~(=XPv^(Xy4S)"
    "6P!33{71LFZnWc*)zBWWadB>uGREKdmqrU=``Kg4g)4){+s^OJ`;_P$p}~^J+Hb%}<i^x4cb+w6?w)L9PEmG{Q`z)7LBnu=&wglqLR%-;1$>`QkT~BG#X#^{"
    "C5q(|8277g8I07mPJr!Zb-eU~Na*!o!LuZ(a$8?hgDn?TZx)KG?Hj#6M>i2=TM=Dhtg4G{*xpVvnbr@su8@}<j>@?t6G4QUHnXt-2W4Ntrc4~IT5{rI(?#4m"
    "$9rh&B2PSgSzzxD9u_SVV|MD|oZWsJ(6{A|fgl=0=e;^+W9YHcXriM`5@b`OIpl=H%%upLt@jT!w&`&b!ERbuD@t7k83VbH13i9F<_T>{2+;N-%s%3S7NP)K"
    "kb8N0#j8gS)?c+^#ZVtD;h_-4Uk1=_LQ^xDPc`%n0&(zmZUvYDrkDuD<EJwL`#9JEUE@#h^D+V5-<c*+(P<{S0w`OUMl@WikduMa6?7z#1qnVz9;xiL*<>g+"
    "x-<yW+_1M~Kxt=*_+3Y40j1WXgOo;-39u<0X$Ilpv;F$|ioNABy9i`IF05xqBEX)&lL@N=Boom8quuO-5@KczOTjYCY$~ZH)4RUPO}K?QD6v7XU{RKTQgYbP"
    "dvJ=fhkLPWs>9MV3A>}N?*LD>q@3iFHG;_MYflEs#y;RxOYFK>Air<Z)#b-`J)(MsItuaqMQV&3W#af&`#fu4SGraZVS9b1E}B_6D6}ydm*3j<oFOK}Brg=2"
    "QwmB9x1&^<w@g+M^}qP<lJ8c=t^?!+klgG3x8CUq;RCV7Q%hJBp1~>QNX#fO^>?G@_hu1L;j5~i)Q<A-wq}oyNk2*=KoD8sdL%h2JxpL23=i^^{BUI&g*Nk6"
    "U~u;sB#C20GmK%GS);MkqEZ4u4NH)?9W5?!Lc357qJY_wXsF`KnQX>ZvEwTz5|p{bpnh%A$`-g<Q(cliz1iy2L?UF#HBj^gJf?>Bo$SZWD(mTGk3=&{*47;o"
    "P{?{!Ey%khqqWR(c%fD9q2y$9-@%_gT>XhHou26@<RjyI9;Iglv>o$|TsWYZ%$n?w#~lJ0%r#o<FOfm_SGZ9C%OAdCB)b{3Ko;F<-RG=ClOs~k3J?VPk8LY)"
    "RiyRVWWp{aCP%mDqw4c2(OXGmKe^F$v5OfwiY+>jG6ipKWs>wTcgDx_bQp)_$&x9!WDjoKT*sql2%av-^$!^LBqrAaOeEcC1~`BP4H`v#plm#t(@CuoFhua~"
    "W86`*6XpwKEhS39SWARfySBsE_kqG-PbqI>22HiY3|;^l)P#|{!&h_SM~AkCQC-wvet4MVHEKCs(RK%t0P{4WR$v%pwz}O5Ql-5jZcB%dUf<58omGwxd`PrF"
    "HPo;od80TDJVlv%nVC}$AFM4w+E~W+u;0uMrQB?SSR>lI1%@oUe?t$UfCux#6L2r#0s~?do#?NtA3C84c1^Y2^>su6oNyS{6{;rv`yYAp0I0@@eMCtD@Jj=D"
    "fnPHae%p2Xn{F@;r#gGo-LzrFGI#gEwDI;tojl+nmrnk+RJ#F>mu1L0+2ihi)|GNL`5W&<9Av-y@J9!csuC_VY$K1>tRf5w%!~8N3H!jRmK0?4hnYbfa7qc-"
    "%_nMTsTiPFI3-{?2OcS)ghdpGk_e^ca@-CzYC3}@K(;v@bj@R=Y2bN`Z$q+a0~3g3{koVfb4KK#n!2<oX2SVCZ>jw_8cQNmiH2U_=EpqoL=Z}IRsM#xT|wI&"
    "){d!dfb-JHBYIr&wr6Ntj0ZspVYZ>e0Yae8;EuhW0Yj)<=x`|%N{H4&qPjCUzbTi89v$Q*TLd=HilTT1OPbB-cRGf(@#17dCPN67MeNm}=o$iOsAP!dR`d?w"
    "G~(^sX|&2f#ya7Xt}`{-K#cX11Tj`;$Btr*zhtRzY0}R6^G`xDO<H!WQ&k$%|IESz@k-P&$S5L3j9XC{%VL0F#_%B`n(R2~W^t5bE6@`pL55D?$WOsaSy6WE"
    "BQb|QCE%7P1MHEgLQE>C%p*feE5vII5+WLB6fO#hQ$Jfs-R~6dJ-aui$!U}79JCkMoYKrD8qbolh-C~^nvbnj(VrxWBx@vk-oVJFAwb}mUaBI-36VUW6roRy"
    "zf!zDWoa&_763+KvgrNZ1Nv3U+t}<?R>#`jrt+a?B=e!QBzy3$wPL(#aKzSK%nm!3)@>oz%`_dC5$3h%8cMo?Pe+?ZPA*}LBt1yj+3bF&#g3W%<N4OSJc&;O"
    "#R@Z3dS^r6LsE9yptu~I4NN@F&nMPus-@ZK#J|ngBoV`&56l2mlBa+}FsDum&XRM%`eHguYv<}FxwMo&j4b`zwKn%Z^m@D-4_b*C2C;DlaVbeo8_=_Pnn2I`"
    "DFHp;hYp*h5%XC;5av@PtMKSj`m~5+bTCiIt!btmPrJ|cxht=7M?#e7(w9#N_95*!5QBdp_MqBpFU53O7o|EfO2j=TGirwR+2Pl+fsTPlk>*oe8FzYj4q}ku"
    "E27R}ERskqSMbhYz#F$7q^=f>l+2VDLQHP#WWYTyxk#s+BZrd_j;rR#5TRWZ@HB@GML2b0J2P>yhc!_MWps6T+g9MFh~dtPk4klf42p${UG>5pZH8twMI5pt"
    "z;onvsWOI`MvMC8hwsCXZAp=KD4S~+AcB;T<9rE8qkvV^_Ao=b5>lTm-!apRbw*-?%Be=fEK(r5&xTgofms@<E#K;p*8&Ykx<7qE3A(N##flZ@q6?b3IZ+R2"
    "&l;tN)}CgeZyd<ybsu17{?bwM#`C3d9sLfQ<=T%9@+8|^#C;wa+R>YA*IbGeB7z4?B0C_*tUyzO@n@40y=Y=W5S!r}9LBec%0gsEE!02>Jv864W(;}zm6*S7"
    "`vC|i5n2N-L9KMGp$?(W@SvO_nHCn|4<M;e7{|<dv;T>VGXglcL82TMR>06eWk<snQvlGR_i!Uw1-hshA~V>=u5AKEG?f<z7mI8V+^gHbX~!Kr(zL@76y!#>"
    "bxb{=FLYZ~Tx1R;?Q&UxWhb?mF5m;eN@*P?UqKIStJ$n-jVMb6H9n|fEfHyQrj(FgXp+DL5;EW3WQh06mOGJ#NI=7Iv$ir4!-N^nNK#UIm@9hC1Sag<QhQx*"
    "wlh;G-Of%P^l3KRn@u7;y=o;6Eu2gtEr41x?c&|Ah2`mzt8blIeUw#GvhS{y;yf>}JrN|**jus8o0r?H)Z~lg%+m69kFHF$^qvMd9C@0yl%O?d&j5@%%hr8="
    "USQPOQGih;OPmLWqF8bgwjTfw42dFD0vpUX2>O&DJ0)%$Jzhv334I%CN#GoaP8GoVO~QAbVH|oYse?NmW+Vbem=O~%gsOT=fw5|E)dn`6c4u0~028s<4FJ?E"
    "p^5gZmVyW&%_!|g6V`?j&oQ;Af-q4k^;D>ED}va?h^bVU7n*5JgoBFcRx9e;FfEj?$1kM8cdnH%brq>ii?G7Pt8HDhEMUXT6s4haj0pk;dpNOx3XErUqPE$u"
    "c3<uyFTG93Hk^fo{xxP*wM?F(&{0_*s@)9ze{sdAb23c{R8E}29xgRot(-v5IZIG^)R{r5vAro_z+lqnFLUDa*`jQ$1~OfB%1)bzTq3s&%A5CdB-kgl&);ba"
    ">rwaCk7&c;w6gN*nXyQ0(xysXlZ%L$=CsxkK0TFeOjbM6l6H9$p`+%K1-M(7o>#!ULYb&XA(nBhL=_M{41$Nj$XG)`O$>p2=rNdOd~%TUjHp%%VJINjYm*KF"
    "Mc0sn3?Ihx1C0t~zF<UzBsW0Z;!lgAB$664L!!D$uok=6S{8Lp@~R7=b=M}Wv(vL(Mgnn`vpei>&u{1zEm4JhXP+!|hMm?3tZuR`5MphMz%m{H5Pwk=Q@vE6"
    "aBbX)7<6UmpU5t80qJ~YIh(;U*jvf1Pp1++=$fyS_&8c%jvq?+3eH>X=oX4?QLaolmYQp^z<d;?d!Tt%E{XEj!v@<YvYKtYn7Bjzq({61eXvEKMwa*YwKce0"
    "@^GpgE^+pMN!-}xf7MXoi6R4(A;^@s^pYc%WCKXct4gOaK}*`(fIpXbtBi`j0ZT?+Y|wZy33dt+&n&cpb5_#*Fr)r4_d0r<P>gF%Bkb{pp@hWnjges?TQ*%+"
    "KXA(y<MTR}9Zt(xTac+p7Fz-C+KX`UsDz6nko4a}xEOS}z#0=xksk<YDpVxbyt5@!@&#cgT0hIC{2`(QymDk$Dw=EU3W8VBtw~&zeOP0hbOWZN#Pi!qm9rDH"
    "L+Teq9aw_wiw=He$a6Bf9R(suIP6Ml=(*6##RP1~68SI{t1EjLL>@Aok*3n3!*{~)oS54mDgv5e*8KQGYpWLPeH#_WH7@W7>X&>i%a_6@X|_wk&AnEnis6=w"
    "gRJt8B6VKEwOxB-(!r(+1+4#o+1BF<Vw@cU0h~xg;r56OxzcWfQeIP>>Vq!ER1QjDF;$cBu~G~6^kLS}39=Lq9@=ccsDzyv<Fl`Fl)Wkf4pg43h8hEl{C8+s"
    "{!d0l8%^m`Pe=#2gd31&vN=`&x_R^BBJ~2s94i>Z%aU<BDZPGo_N{s(Xbgj8RBggTqA}Lz#a^OE0EXn%R|+j_Wd|c#kFO0&F<)|M%vQTLCJY&{j^<w#(HFKZ"
    "5=nmObrpgbbw_ZL&xV8{4IvPVStcqLg<Y_-0hTUJq7a|fTulNDp$7y5AuoC?QfH~Ay_)h;PLxe6l-y2&^A%N~<EBm<LxChmR+zNi*4QeF?U|=>Occ;#yd_C`"
    "%}RIf;@XP2vCB}wXa!94ukI>oB7Jb(B8TH>{lC(fY6?_Jqv36lfQDlENzIi=HQ?5i1T}G?<m{POLWS!uC<qcjel24-!y|gYW7H0ux^jv48C164pXFj95#yNM"
    "gVM6N45}kEQY)?XbPcw_ne(howI>xs>0cS^djDT+2_-@N!B)ej3!um1A8_UeHqoAIZzUmD4a6hKNuk`kGDG2Qw;l*W%}<jF5@b`@FZ-BpLrxG3@ktrG1pHv2"
    "$sYMGrP<gFUKDPz^;JoHt@MYNztVTac)n0_@5)vx-GJ82-r}J~vk=LW!D&`gO`a0fMm+M`r!uT9y@&@HV9onSbDM(<Z6?c6^V_0FPN#c#e8cYL*EF+?mvGZG"
    "trNU9v9_4#lGWRxX_e>+EZt5lvb^b_RnVN{IYar+Pan^A0Av!rB;y|QUDwwQO;hpZr9%+_%6Zf3bZrF2NZy2r<9oc?eS9Qt^vTicmnjX>1#BsMpTx{C3m+97"
    "%6p%sdeePo(k@I%3#v|2!^%Fi*qTv~zNKcSh@wq=CH7VmuN(yJwS%}_fH%#mi`R80UvUzRt)2lrX;pK7T7;g|1jm_^Oo8K8kI7E*!L@*{L>bMGrx&Z#kw0cz"
    "BJlFJZpzKRL=1xL;7tSw?%_N|s}9Xf&Yq~*bBG6?)S8GOZ=HO47^aBkBfttBBq_m)+Gb7E*BR>}dOIzrc#xq3Z^`OF9kCihLJ$J{;0-pOo{;Gk(pF^S`XJs^"
    "AO%{vr)bQgz|4WK4O|p{h+I@Ufg?d0zEH>$T~Q?J^w1JIg`yM0bB|lQlYy~S0TTx!ToQQyE6&}**;64BDFIjUt?BW%i<>TMpaQrYl6W>tYr{w~<E*n<Vv+x4"
    "%tfZ;Z|`u|tZ2l6jN1q=m?-;?oV^n|fJWO17a}Wlp`xFhRupGJxrNBJSgKtWDQP#?JT6OOnn(iA1;d{EM|Dznft2Am1YVy1=4rxb_ZUe`6@6F1UgmgMOufu@"
    "^t5S7`cl$XJA!%$#iwAl)1xoUPse7yBF6;VI0rj6^S&<;me^ZYr>QMmuZ`8K2a$C^j7#9vMHv`qW}x8#)8M5N1we7rp@f-z8Bt0Ji-}>+`Ut%ey8#=m%7L=D"
    "{rgyDvz@>SpHBsm+|4neono3K&T6n75mh0TSA9m?$bcq1g1S3?aE)87I;c5`Qi%dSGZp5rl;mv*KIzuoksbJH`L8&M!Y*er9Gx#7)nyqlb(08rwGb^DZAdu1"
    "hAiGBO(YjW@evdaXsDtrlkT$#)`o-CU!y<K);N)DCAAKWEAw!sx+oL<>>=*J0E(?k3+*WH4<c-vBIE|?c+-pYa*>5yUfaf8xS+wC2U3+y+w+6lo{OiiGxG=g"
    "_0h`Ae5buyO}&ffqB~<pN*k(mUR=&_&sBmT?$4?l_NKbQWF*ubxuG6<MQdhT;NBBPNHB@?n{XM$*CUXL<jT0D7F<v62UrCL(MVDnWl=ycM+0Vu1QcVU?u6ZE"
    "_ks4HL6dGPBpNx2N2#voi!N#jM<int(u9RYgd5-MiWiRqkY40)BP59=j|jcd0@{sO6FIWssq_2>1`VO9*cIc}ktXCUXTiA4oFL24w4u~eYA4vQQQgFR8hcw("
    "q3sKR>_EdLNO3yJwfGdUcfVFj5R~lGdXimS33&>dKil90z|XEF032aM*fR4iFJ~F=b@;|-HNfbFlIDy|V3zR=ziOIc#Az|K9x&5*TjJX>fYI>^T9avg-86|z"
    "+I+CwcFLh?1$Ah}I@UeCI2f`CXHQKxc>aI~q#3Psno*-!(s7$eGx@;3Bh5&_!k#9{$Va7^p{D;=k4QCY24gkAQ!7@pAPtF=_7$_*aHPnQ)LKv8QU1hfaDvAa"
    "9cq1USV4XO)&`+A=v1-7N?R9Qp#Ty|xdQYCe)#Hp8rxZ<6$ojVU@ozi0Ibkirp4H62;rrZu#-y4FM%Y%r4z#Dnc-*m!Fe}3#R8>Tj9e}+)}w4Bb)NT;rO-7g"
    ";EiNNGvw$w)N1osRLI1TD7G4T;F)7n<gMV&1XvQ+*(Ic`B#fJR%E{LlO8K#@ZZXhMi5wRU0tcMg%Cew_$X+lbc7oksW;jhDW`uS#t=PsnZM6QG039_7uqcus"
    "j&q{0mQ)dg@|)5gPA!d#iqm<pleGI?219@7#^)+s=w;q73gz5q@a{{Mu%}cd?BQ*A9<@kVg@vSGIwBYQ!CiVFZz=DleT;S8$t7{Zr^g~c8H;8W<gwShkl7K="
    "OrjD{Ynr=iH0YuG0EhUrUR|BiB+-E^!eF5v@Kz7W7p-OBS0!^E9XFE-A%*>Iz<HUi#Wdmq1SqH=#8KL4US(sY&fA31S(TdaYGtKvw&q}Q1Y{ydAXV|H?v#9T"
    ";Fj?~vV4J2FLF7)Ufq(4|00^1&p#@ybFJ{!F#ML~MU7rV-z8_X&3$&sWtY|V<R&J;Dr>1NFBud?=?g=Um|>JIZ47Pq#i%)Cd2nBzUd74dB7jOnN?jInjTAsr"
    "GFH}no>f!hgh>igQ5&8>dhL0^GZ`Vd{i<PASr#=TiTO*lD-Lvr?X0}QW)r_=mwV?*+cpoRR(r4-XsM+5Ni%y;)xf-Iko}XVPz5birco2(V>S)~m*Ly|J$`&4"
    "_Ruau(IvRe)me4wg7HYkMW|MTTH~Lq#r4{&#M73^=EV@yDtcbC%li~LBB)r!Bmo_Kt~ZGyK_DDX96vaKT7p59!8}MN0Z0o{jk?edsai)b_V}T649%mnVVFs<"
    "ihQhtBqi({GrQ^Bd|J0F-qsX<4&`4<w4Mo0c^RJFwm@jGjT7tFaTXc2{IMv#!nk1hK;U`iB#HTBGvHoS7XrmW#kxO1sd^@_s@@y4lw3-#<uDJY5;nXRC})V)"
    "S_l7)ndMl<gsY6<_14c5f+(ggqAP?_SU6fTQ!S;=8?o%W;tXV?0li_7Z<)kGC5;1nMsHji(7B`3tk28}n=HDCkaX3H*z-c;d7<$<K^o6XHLX_M;OSJ;0?1BJ"
    "(oT({6JO7yu%oWJCT5GP71RR67G}lVTUWr~y$CFX!asRo)#X81p_VRLQU#i`YwhKWY0<S5DjGC&rn0n}ifpMjtoSb)cggx|pAuZ$c8-O|HMw7nMabk#?BXJT"
    "2a2_K+=GStgERNM27*9+pefgk8P^9YH*aRTvx;{4s`B1pgWr3Tx^8><T?kz`Vj@bcq_)Y}aw$M_34qaDaZZ$UrM;c`_0#<XRC0d)49^}+EXVB@|LVcxQEGGP"
    "h_^h36o2Dwihy_uF6tx5ERZxNA(}z@m-6VPft|^Av#cCXo%O?u+0IK%uYlU~+)H{<Ta9C=;w9d*7tASo5kb0C_+sW9_XlUXAXc@yZ~vlR1XRmc*zRA%D?Tr?"
    "Rz_uNw_8A)RCd6?ScCv{7<jpmR}`IPrXs))y6AV9Cjet5h&PCA3LBth;x>MNhU~POQ5+0#v;^UTm+m=<FZv-poA)MCOm!@K&t#}OT726P;zf=LP4KVo%Pr8i"
    "8QE!=5*6!f0Ar}uBuc7OUoVpXmG~E$CyVZPvQrqDEy5z+P+pCJ+HTL%#hji*WCC#rAV4<tD)M(LB0tRny$6c?*w}?e$>ZF2{k6Uv0&G55qOkt_yo1ea!bSKP"
    "*#~(1kad&=zB%rz;X1U1W{WL_v1}|!Yju&3KK=0Aq)!iBmh16_PtT4a)4L3pjrvrizCYHIDe=U-Y49nZl|=Q&Sg6|ras)QTyL~kU9j{_?pDe!>TzqRf`(A<@"
    "T9}IxnTa_}W`TBh;vtL@o7jc47zL$az)k1NX}iQV&e6!I@g99t;K)AcU4#aSw8L1o7J<u^Qc~zOX9IcqPUY1tW~G{<DkAN!P8v5Ug7^ZEGAJgcS2|IuIEz#-"
    "0&_H2bQ83i!4MAUMzA~2Cj`2t1hu$Gc{+`nVX-@A=TNwY+ti^iU@1kbgmIX`?@lVDqac@xCh@vsI7O~R=jWLcOIkH>N{&(tcX_p6ft0hI!!Q~lE?P4PSZt#k"
    "*-F1oO*C!p*jQK%&qL6ED$@rsC)yOvP)=goGz>M6O9k&x)^2t)RT$@@=IXVzOkSPo3P4ot?ToBv$zXX3NcE!j$dwk)o(0?Ij1=8FGPVzgU-b0x?C3~7v>vBO"
    "+c-jIkHQ%_9s&=*3}ju5NY`47#3O-!fop2VGTr1}T~{SxY)KmNZj>FSQt?h`Cyg!-uT0T#UvQ~n=~1onB50H%=%f^Zl*SCPoC)`2A({?Fv{jv(PRwC^1$HKY"
    "b#dQ=prBSyrPeN09*a&cA@KA8W=Rr^G*6+?tIeJi%`TCRc?XonwTn=;o>p<9e`uI9u|m$BT(iTI9+-;+*5CvI$qi`0jbH;x4Pvy)OwDR4yOdSe9oiwK$WxX8"
    "l5Uti5oohQwi1<gMP!~_t8@w#D~CwW5?iU={*&uan0H`qV$4iTgn{YI9u&E7@QKx?zf`G0S$SA`jdpW-)Et_|TtDb#rx<HYt?5u6yI3_jNvm(F`{p#=#<d^?"
    "(mno?!A`owb#<gwzrtnAQpAha@akXTj4W1!hLyB-%IpMj->yUCNti5wNkQsmK+}%~9F9c-f?_5fV_wUkM;&Bh5n>lY6miAz99frP<r3yANLVNGxYf>tLba?N"
    "EzYhK5n^VZk)Z|ppO)e<`=)o<aJnN^-f4-Io7o)96u=YN5Ft*AC`9~*wVQlSOkj*Lb$|F;652xV{#~}}Nb!IGfG~4?&Dixa*BjP4V*{&RyDI&eIgm!47h<V7"
    "X|Ve?o6;nSy%oQ9Shmd+gePsL9P@IM?WsW)M+`b(Ntad;s%@FiI}$$5;uUW^Yyv2MHfI3`116h+YD&Ku%RX3>Yic1Ve-r5ajw4t+9+b8lUnEPy_tc$mkQ`*k"
    "B2kN^ttIsCK{DgECqv-+*K&Krh5bI(&BU#DXBB-u<VG@EZ}E-z{aOH4Sev6JcR!382^oOYj=?+B8ERvmk5?sa0BmVX8&FU}0DOS*4--pRg7jjh+9+_9WM8^V"
    "G%8#L={AKACr1Tw&0qS_v#HNMTDaTOTW8w>V#ZSI#_Qywsg0y*3r(IfDr7p9;jzryX*3$PQE8g~AvHWNkM8~Gum2DAa-X0"
)
_V11_RADIANT_ALPHA_SCHEDULE = json.loads(
    zlib.decompress(base64.b85decode(_V11_RADIANT_ALPHA_SCHEDULE_B85)).decode()
)

# Current leader syouya tobita's stable executor from episode 89511601.  A
# second chronological source (89512693) matches it for 717/719 actions; its
# only material difference is the late shared-market sale order at step 624.
_V12_SYOUYA_SCHEDULE_B85 = (
    "c-rk<O>Z1oa{Mnm^PujgDAG5M)N2XLkpzlzV?7WC19%Ms#`-Y!&G3J>Mr>ABRYpce=6gjpIlMLa(e=J7GhRgGFaLY;@4x;2kH7zZ@=w2<e7^ef"
    "<>cGx$$$LzU;p;s-+%D^<3E1;{XhTyKi@z9a`OJe&G!3mM<0Ir`nO-NK7RQ5>iXpL<n5>3$?59;$6vPFci(^b^Y;4F$CK0Z*^hVcw^z4+{P^4d"
    "osJvu)7KBz?|%K>jN7k29x=4!my_N0%k7Vce*AQE`~KwHv~7QU_vf2W@4mjh9e;hUvHO2}J#W?L4{!ha_4DYzM$Ou{=5+k&uBk!;*Iv8k4Y<B~"
    "yWRhG^7Q$5MXmU<y}mv?e(1l@urKeo+s~+J_hWPT&r$oneEPa?jcw`dC$)xd4pTg9*kAsb(@A5d)3~0F*7j<@yjq6MADv<N#tnLVWhQOE1Mfdn"
    "XK&Vw`|pSA_WR=ccsp!W)Ew4|wLTc271M?Neekbtw>RlF2<v!y9Jw3pZkq5U4F}L+QFk34yI@BTB69@K!}3zucNcA&ahK3U?VDR04%3kRaKj}#"
    "OvUg4&4l)x`=P!2{p5lB89~GEP6+S)ypQ`)gmZov;?XDd&9aGy(+=Y71MA8rF^uy~u8ijF>q<SggF8$p9NW83sb1SrwZ{d0v%s#LB?kAf1I!o>"
    "52k}T?ctd)iic$~jNQC}&S+}A1@8%P_VP^;^op<6Sy${&?7^(BQ)z_3@-#iqb(n-{Qln2z{2#u)c*G7~4EzB+(j4~P+fUcm+qbvB{$+b}`{DY-"
    "zfKlODt*|rwh6qj{B%KmlNU!N7#^)P#omX{99lOVSs<3kbsRr(b{4ClcP5;Eck}6U;Ss|`k6LU42C&HvXYl=(qw|?PxE#IDqAwDdH9cACwEG51"
    "&FuXl%R4!H^YrN@dbI7qXrC6kM}oj&?<!N()`$HT#%Yr61o&lrj^_9n?7`uQ(zC-y9KyJzo<KWMnIokWhwrt&JI+3|ud~V{hSBLVO!iBfzF&?i"
    "=i%w8^S<jIX3F8L2ox<CgzCuAF~p@Jy7H{TQ;K!<h)=<MC|FnU=v9RK_-IHm944p)U1?1Lh7Xtpa6zI!%7?pyw0ERSkbvBDu^9J|AK4M)uy=<b"
    "cX&7o5NB_vRFHdUIp`m_0xGOR^a#83^e5OTclh5!f@!~(Jive9R%2ff(|K#<NP|WeTK9<=hsml2?{M0L$(eoS0$IajJmHvv)zu>gt1nUMOH>M?"
    "(lE4X15H^_nh%Jbs1&^3&?gjRDF~_rV=fn?GMKGb$nAr%`Rnprh)Qqio<W34A<RSvS%?Z&NbeAp-4rBxL1I)Eu+bQG(O6U;y<#ld_W>`l=u0f>"
    "2A&{BeThY1V$p~hxOgzyEfP-jQz5W5LZt7nZvG|u40t&eWD%WQF}HkrZ?VD5_Dy^!c%2)la~cCH40wE+6@rR~ryqVaG|nB1FSj>WyPvi<H-EE4"
    "V4ZaeL!*6DM^A9#Z19I}d);WqC##`7VB_N4B4v!f@i&bY!uE^DlnYk|kGGxgoA)WvIYNUakF`GmBas_ZyWDx!l(~DdkvT=#K~81U>jVwM{XYAl"
    "^$BgAU>ERxIzi%mOB4gaYn3RLOJLj|b<1F+u5|)zH>=~N7eqp@2MeAhNtN6Bni_1msCu(dRBhkr{W-dcDBFtY3S(7Wbi?*`n#r_&uyuvJ>~K`h"
    "C7B2!)U=t64LB(K0ybshXw{Mv7n?5P);ZooTNioa;mZPhckr-inHaNE7w7Eu(}2D$cMJs4AUf~WF&jgVl|~aCWs)G98qFam9A+*>&}_Ybps`Jl"
    "n+SH(!dg-4GRPRng&gSdgECKOOG1FQ7h(1hAG8n!*n-^4(<{y&Iaoh$#fqUmTEasiioXn?-GruQGM{Sb8wBFu?c545157ayipNi90`_sR1G>ha"
    "-sfcky1z3`qN3AGbOlhhFpX%qR3Rq=rz_}4A`23Hj671=YqQBvYIJE3rnzBn$$-+%67jo^%mPZSM+Yg5CKF&&I?@cn!)N=aPuJ`%m)S)i`*C4C"
    "I}!o*1fEP-6(E^_{vYjTACwR?Ygh`FVP;cFHJRS^Rc^v9%t46_f(47R{F9QyhTelyls(*wT~i&Fo=Mmpb$thTvL)ptpR5r?R^NIuP&W1f=Pj}8"
    "VuAd=P1jc+-}Q*<8R{s+_pei9<R}xzx7ue}1H00-f(YB|GxfTem4iYXlX3a2ZO<8ELQL{Pp*f|X#Be)Gm3hl#6;c0-zn6TsGIkvxFM#A;_s@E#"
    "Cxj2g7Edi<QFsQYlp`^tz|`N3n%|p6K!qPw^`v%`hqpC*d`$XL5&?q93fCjaQR!g<!(e!jujGd-(<ro=w*rH^#~?`@Bbs3h%gh>$r52SE2x?e@"
    "#O-KtffL$=au5Z~mPA7pSI%TJwu&8JIgy~uB?k3tlUBCC)tc&(^y$r3rzR31ORj;UFW@mXwC`m1z69FTGNyET*(1@6lC^b*1QfDfRSWVi$!IOJ"
    "9A0RZdnh^C+;{Nj57&QXfJD#q6Y`PqJ&)2e0@{vwMlKvsOlD1X$m0%y4CWdw_Ls<@`zzcifaMQAVkEm6v_KZ!YTf6oM3W;@&k7I(`op%BxGK{6"
    "Y%*aN5|g9b^HKGAmFTS`vY*`Oy4b~x9K{wLNST7SwlYb2m^<U+c{+^4@?^;rT(So@Zm#3eGXzhU<N61TdlHjt0Va}eGy@#Kf(DJEK2SCu%;}_7"
    "2^b=H_c89M*$MLnvX&C1V5}uVt6kgS>-#`qu&0!_F@vVsVFoV%4Qj&3-Qh=b;zx(JhEZMAV19U*<TYwJUD0+2k^u8GqE=uSWVX893sR-MB5q5E"
    "kY3-;rJYrd4}3_pKsD5`B6*`Y4Ln7edzqP24<D>8LE2cx_OL&h9ZI>`1hGc6cMA+zcK?PRLIDrvhbQ1(!UYBdCORP>t2dp{1iPl%?)o~S08Tgz"
    ">k3tq{{5F+J^-pQVjoeG0Q{jrx4^d<2*2&R{jwX3!>P_5bvJETvCQ3lFm1d&Q6~?0$d!}7E!A$o<7FALPWHI_%eqp|CV%6dh=c5RAO6`vq^g7q"
    "4co}0HLD1N0`ua$a>72aswD*({b6Ph2b@v@cJqlES}F$U6;26Q&Vfe?C}9!Bp(H|Sxg57cjhfD236O102VL_RX&QJQ<J*vI+Q0-NS-&o3%bXE8"
    "sHQG0ikWb}&su6fj>eM6RHC65xcM<pJQ0M_T$R6JZCB8ChqYsB8{n*T@`xUnyzLp<7UMxsLYQsnaDWh~Gq_`KXTT6D7dl)Dg%YCmkf`nq&Tq=)"
    "p+^UK$rgbPw4x}U!IEY(`kjtpZM-<ykjW52Wf6NdD7uCK8Y&s0xfQ(wIE{Gwb{efRkg-nqr0YzLHV|X|BteYTt7Ato#y@1KZ)wtA_2-|2WSX?>"
    "Sf{EqrvI6R2jZ2eV~|lqiWs+|FqXvt!HnTUL^RoP(#_&1$5x;xNP-NVz>%MVm9nDj*hgXxeM-PBPX^c{QH7XPP?<-Dlvaq>7$ihA&L~_I5~qH)"
    "kh<R~-g|a$Oq0_l)j4P{usNlfO*Eb*Wf995s5Bp2tD=8N6iL=d^t^$QO+$ddF}+kpj1wYxJSjq-7=NXBeag~YPAve8#AMO?y$AGp%G=oNRaVE^"
    "-lp=QW+d~WwIqA+ueD;lYH-BXUCa(Um)30|*UdB?ml5W*=o+fYLMMbilSfW&VT>j{$k^HR{@}%q1>@rh*UUVLQ3K!XnL53*G4MetJ8@86uBlGb"
    "Ru2?_Kul4c`?vX)<YGi&0rO;IZlO#Omy;`nhfBr^23+AFLjY;1DQh$``Cm74Q0}W7UyQfsK|KLR>=GynmlEl;K|h<P3H_{}67&PU=*LD_cR+#G"
    "4}=00=_@?mls+xuAsx&U@^6}{$<uyxeQwXIY?2T+y7C1Tf}uz|GQ<EOh(@S}+e>j@)<yA-j2m&+$&9|CeRlXoZJ=x*;-vW$SH_**rh{l?_>px<"
    "VfRHAf}}j8R07YSIBrHreJ&VHS=M4uiZ;`ZfV*IF+0L9N9oj}X(3)vO0EkiVb9)N4krqHLCt~Xu9&G}~kN0i`g^GyptR^Ws>xi6=6~XF-M_Ob@"
    "<N<P`OH7D5&rpfj2Bb#%A>T4h(tr}Wb}=GI201k7=Dt}ThjIY2BZe||v{_<EX;ZtIwkEMi!r4p9i(<51E90O+k+gXPSQ4x0wY8iV!jP%Du0>jr"
    "PQ?mw@vsyCRW7!!xoh?i)`w5J0BFu-AIxSRQYb0m+0u}Weqqhh=tswIlI<-*I**Ln=uNh3PKXXE2Jq0J4M-n#)QNd}Hu)S?9ZLupGyFv1Udc6w"
    "m#E4>A0v5@M28JAW{A1!wi_UZk8l_}EiJU9@YPG$9F!jvZ3<CU!cyN#iNSrFGd2idFzp^8X)$EiG_bW{gEhp1sx%m{z}3_gtM(F-yB%A}og~pN"
    "dtJ!H8SeFL1P;4PnY8V&R)SQ>VP{#XK;k~Yt{3ODb6EmOZ`?3x2(_07@RX4TSXz#s-7HKPBRE)Ahl(O<8DY*GXw&9LNzHG0&&8aOjP-sMa~IMC"
    "U<8yHnIv3gD@-`?j2<QBin(&v4fCaTyxwwWs(^EHdG_q`JD{>`7U@}5t8S<vf10s1ny<TIi?pnsN<}g|C2J2MTBP3W7FS_sCb*jHq`K(16f8MC"
    "vACNT+4Q@UW(U!Q{Kl>bB6xnayvq$ooTe?ka9jEeFsL(5nSDMGlxk>vb`+pg$p@!^NjaX!jpUV3kqshg?KW925eTjX3FwqaS11ifq6XEK@rh*t"
    "29J$(?f`18Il4*Mt^*DLyE<pg2{1evWNj70It2Ux011e*?CPJs<O_Et>nxOm+A!}{0(7QyE&e@~-pQ7u1kIuREp{(Wfwt*8C}IC?yTmImZG{@B"
    "Y*HviQ!V(AsZu0*J7|a^vY_&F7RzDJ&kqx$p{}{cJ7lr3QD`c0qFm5RlC8k%o58Tuy;YwWVaZV?Blu+^9*|&;P90;+6%X#%+pyX95GTDn;B@UI"
    "5E?L6vnDY@LF!tU&N3LAsJPA2h#2REbD3(H;u@<cO?MfrDl)n0qz-uAu+o(&on9WCCbNH`+YHUJn|2NxMxB<;ie<|yec~_bC>Yto!6)KjME<h@"
    ";HjgQoC8ZBNvaDZ@8Vw5*uoQ=TtCDnMQ2SEvI#XLf$6P_=wl5S(G7#jRI*n@A-yP|l|W7^sGfOqJ*+8I!PyEf$!u+}h1p~ng)B%TxDk{t1>9|y"
    "kpimDR{$hqu#;78^-==*rB__7nfF<?@m^RFDACWrAP0t(2T9rktg@m;$u1^I1!j+p#muo$h@w-|Sk6ITdL<5BfLhXR%^*I&1F##M+m&PKQppA!"
    "pu^}B_%kX;QZ$s6A}|yu;GZmxrF(~Y%P3t24zi5(n=Fnp2ok>rR-R^7MggxF`R>NF6j9EtOEcE*{_M$TWsDR-8U7sAn~2HHQt}CK=l$ssZ#qVT"
    "0Tj%*=Crzs5`s<FOUv?L5hsx-*sL^+pMkpjyRre<*n^H}IyQe-=Olp4j#?T?rYR;!cd?Z_-$e`nA4o12gPt;b^00~(Djw^W9Yo30Nn0uzKlrN2"
    "lRF9<Ghm49H~!hQ<}@dE&A4PQ0>Yyb5CSK&e(bG6>F=fWgF$}_q^;w^dkR&+nKmZJ)>;tWp(U$8MouoT3-AuTJm0B8(@F~7_MLr5Ta<>yqI~RV"
    "%+^#m6D+r;FdO5^QIDbuY|}trie?_THmR7JNb{;J8l_0MUYkI?VgmyZN3M2xgF}PnARLm4<}YIhl61g{59<g+2vWl;i$xJ^Bz^FAeeAkv(xJ3X"
    "zDQZ7q!QQF!yoRp{`T>?B{>ajxUVf^<&ah}`a~dN+JiZ|r;WFU#WLqAO^hrlVg(izxp(nUJ0aXwieiYgV*p)a89LSU(lKZ7B6pVpe>PKC&G4Pv"
    "Qs;nvfR#&jhBII#7pBH+X1S)PK+?&z%|du!6A1iTi6fVo<)TjR(!*C?Dfr5I-Nc!BK}{YKHQ_a^o}cRy?=QDs($n833Y*Y8>>>S^dH^RntIb~F"
    "DK43!3?W=AIPRug2(fDqfSIsj@x*{>SMbX!LMcqvCd<a0f_yDS9oD6CvrgUglCeL#T<;2?09Y&0V6`S!C66Z2hgeKqDQZSM5sD<XRjNXd8oIOU"
    "SOSF6hXkg1&_wn4Si5zi295NopwF=@XM8mC66%VxaDlj1Xghi4Bx?IHfnt#V`lZapB6V%xrOo#=ld1yaq5;TEXZxHWrGeVI33lRCbM|}MElfNK"
    "XVY5+R2Y+tyAQIf%>>he2~925$Qv%gXb|qG8W4&vTz+fBXTOS3cE(sP7&&E@E?zv4IB9XLrKSLQx)#@9!L|V`dIa_uuGXFegnxtak+(R7lHjoU"
    "-z=v~lB}%5g^eixs%5d7u(e2(><6WMS$5i5B}(4<t4v*#itozI3Wj2+1>bTCDpe5|FvdG-lu$rPJc|;yvlVQkrdeVvG^&~tZU#$sO2!3>2b1*9"
    "od$%{od7sfmF7d1JF4=ujZ<o`sglvd60YPJSd?*68u%(EM4hlERtJrmnvha{1Zhc$ew-R|(XH&j0zk(*9^JXLB}eT)OIM9o=n^0a26{q`poC?6"
    "oa7OxiO%F|*8UL?09w+eg@AZc`0B7kII&jR1Kd$BHT>lr&3v&)t&L!b#8+6kiyylx*vl74&$I$>Nk_9EEBm5p+cI_70cvDQGtrzzFG5R6O~`2c"
    "zHuH>C+^l^)`;03KRUqhZWSk+eBDUi`~E~7exQ!wql-WXWo)Ki_&YEBooB}13G-y7klNFbJppnH0bP!hzytB=1aV+_5|2il(zq9O{H`o>f&_bb"
    "g$<A*S#g#l=SX+V4jm_V;v=8*PL@Ivl_w?6U~y^}sf)zgfhJ<ih|0l8-Q*`n)zSQ7xLpoKASAr8^8yLrYZ`J#KwXDakq~y^(|M4Oe(Nn2?r>B!"
    "#1F<6|2oj1(9A1Z=W-oGjUS~cp?g=b7{AsdoK-7Z<?ZEM0~2o9VkLBhDflU(Se0t(<615x7^K?dz?)ps3dO3lB$w3))Cy!Yr0YFUJc5aFmI)cd"
    "Rw5c5L9jaAx?&;tPVkrUf<6D1sw^tG6fr;Pz*KAj6B@(c@qm``enpiK-}C`mK#qGHSPLn<EK@0-ONIJg)vq3fz|-AfCNEf6n#NPR6pD`4yv0n9"
    "zAzUVVq9n^6q&!~ZfGdLqFdtz6#Ik^$HUspFdJSw6HJokn!gr9biqy=8A|J53jQW}9jT*FTt2m8_4MvrvdQSfhC-AUk#vo-RA&N$dG7-Xu=+~W"
    "$w;~>?*d@!>0tn!M)IF+jPo@Bcii(>GyTbwYWHD>)IGf?*9t~hl?76HZO$_|ELl%rMrG|LU6%;}$04Wy)1nM4yPg{mED;Ni0GD732J}6cxXO$T"
    "@!Cidp6NxmGf;sjU8FD^ZD7_SDnZIX1U1@rar`vvT|=y(<PwHyF}GueXvBGh!q;JvFFIg!gfl#;{-De1wS&5>Jec$nz%G|BvZ+6zY)W&4dxo+p"
    "QBA7QcspVgetRtpA^fdg8WmT<>XeOk(NZPcb+=MkqHfwMjy3@tgCS8kQ|;|(9F3|{s;X4hsc+e;O{T)G#hQY+jhkAyMR#-1lbCq2qEvL1_whGo"
    "4YwbLP_=LcqV%SEVO*u5$ccDDGAQCCNnn|{AG3oJ{1HBfJk(=;vgGc&DL^Dw#EZ@*<~LG<7GmaNgtsza?`{{-wO(P^1-PB#(c>(Dik~oPrLP*3"
    "t3XlHo)*GMB2{-?SjcQM$t=UBsvJ{}fH9gQy8xV(>>^3)#&z^il&*wzqb-x5|GFnHgE|~I7lh|CIgfB<L|Q10v<D7sihk2kOqr(Exb?VtB7CZA"
    "bcFIGtdi3(&jiysGwP-bP=-mT+nhgRT@#ap{u${u0u}aYl5J8r4;Uo#Z^sP*rI1bWx=oUJFNpYvR3jG0vJz0aJgN)5X!=3{Ar*CHEWy{wH>0#3"
    "I56Sm4yZ@vD$*neBhE?;DhX9=(u=J`{tw7RthCuK;n(!Tz)%=$A&aC!K0DkMb8b<A+{gAcL}w^2%dup*2RKuphb^$|ofae*TG^>M#)Vq}LI+@T"
    "G8S6waLOE*6Ukec#>H()sa>-rjkTga!2kuUmU|P_!a#aRA$z$~y|Rker^k^RD0_Ubt)I&)$Z1K{5<Oy<2c%W&-$oTqZgBYjWwpM#ISwP1#F|VG"
    "1B(soTmaIksZ6>Mxl+-Q-TC4+rguu%ZG|K^8XMbs<5K&W$ry!rt8|=|N5?sH5jV$%=EZL2ITYDGaM_-T-Hcjf+lMj)J7Wc#IR;0X%(bNMu*0s%"
    "AIqPHVMRnPrI(yiylG0v48&oab}_5wjPXj2Vy{&oPq-V4((&!XpfohKW;6sKnL)y^ewDF6P~xl~<m}=tDZ##{P&jB)qORBuXbQWCsiLl%6yYW~"
    "Ilas~9OFEI+1VMJX2h~-49)$O<CHiYah0D)!idMY>+ohPqch4cO*tKJk2)sFHY!{WUd|jjGk0<Dw(Z9yLG4jZeRdcjBAw!CJ%N}|0>K8hBJDHd"
    "a5Y6*M~1$^eL}Pn;$YSiq9U0H#8il)Wll`{03;Qu@;Uc<A@=LLt%{9MVv~Vx=3=IzJA5FvW(CofY8*{S@cSklZpi(bt$-eDxQ+U=X4b@A@uMS@"
    "a|wC$Ghz0CSfpT{WB=tTD7vbY7@a_&0xC_+9&`gErB<m`-a_I^cF3Y4w92>vfwV$hDIXVBI;m6wDg5`Sdczd6t8gzQ0~BVTBM&&cV^<PK*?E>#"
    "dXo?ce1TPh9yaw#M8cAw7A`^hOOcjd)l0TYO;s8|(<^d%6<@aiayOW#S%`=xNzG$JWi*g_g#sE5?rc-22EA?`MKUVf0<A!4cD18iD3&?W8|+5Q"
    "*nHn<^2zuq_M8oCh%=b2(^P_m314n@T1szLrEvDrQ@TzfKY%?-*Rn;dq#Da&R%r)fav|GPAdNAD0lz6^HN~dk6o;=Z@lC3JRKjUfN44q7mvW5L"
    "I{qq^@s8@h!^x_3CNWc5XS~;Q?4=%22$fn|#u96hsiTuat?hXjN%V6I&71107D(&C7b>V+`n8dmza|K8ZMAw6R8OjddEYWNRRqPs?~6CxNM&+R"
    "vU&zQGZt?gnfNdGpOOL8>Ji{VUd}9{SgAUW{xmX%e<b1&ff?2bLdYV53>plDKA{=K5+zP>n4=WDi?EYcN9VIlp?FK60<l8Gdt=aAK!>S9gJ8iZ"
    "^}j^=OP!Dup@P5wp3yfLV+?#%h3KgmHtB`?iYD@EsMpXfMf#vg6VWH=q8sq3TA&KObD^p*XF=Qi;?GPBFL4$m01;!4+2L5IM0d-Qq|9pR@(?>I"
    "9a5%hTePi5RHePH5By4m|9N}?AgVqmuT}&@D@?+xn)U1zb)zTKLnLyVlf#_6)2TnKs2Z}VW+v)o_G^ibBg3cbZa6ZER;Iez2pO?pyrd_rjfK`{"
    "02!+{SYQp9;|S8RH0b<6`yv#vpvLDC=H0p!sV8|rX`n^(Tp6aSzRY*94SZY4Mp>&@AYNN{!FoE2HP=gnXfGaGp7>F+a(IFT>llSM<fLiu>1BPR"
    "bPSL>5j&^*qm`=zu$q8;?u2snp}$9^VBPTT_pf(KRZ%O?hd@N{Vs+$g58$?NkaH!Xm{QSxz?IIqvu`wc5RUC!-8}+_iKABv*lj5>Cft6)dDzXp"
    "bdrT90${vzVLC}>NibTe+H@yzCD<F`P5*ulR2yUj7Ie}s>X#JEcv;0Y;EF8W_@#hmdv->AnQTq=(2hoHp9l~~epyuES5;uy7YVbCRJpxVP@!iT"
    "I8u8wSOx@^bjqRxAx~Sywb*KoGI!{eb3s%n)?SOMN~(%aSyJZ~pU=gUI(LZFp@mZ7{i<pb9mXs(u-z<sR-|bK`d^Er@D;(BkRs*GVm%aZp1Y?%"
    "V9~qvE6%5ylAitq!pT2N9XwSwxdadD$;&1?(xQ^!hwwg^BxEqe)$MSX!geSJo8<B=+S9DHim-ymzp7EF*m)Qc;3|{B3x97FQzJVC3A1n(kal(u"
    "lED!Bhy32(-&%bHeRds#iDC9|j!UtoeeHt}Mr=A((1&4*1r0J`BV8nNv$WT4y3A%1e0Vp>qlN>$70KpRpJcSIIw`d>i88xT-U_?`K_|Lw5ye#B"
    "5Z?tYDv)BK`4VE3b;7%llklX1f3ZP466RESU^`O9jSn>~7wFZjsBsroS?U*`4~}JI4fA>{r2yO!ksN7yEh)oOl_Zx?O3_+u^wnZ-7%Vg**~Z`f"
    "+j6tW>(+vc5*Hx}*K6FBO}mtz)~982Cw6CM1-6{jSKjA<uh_9KX%60NW)Z)TKyeR}t=@RP^?HH93)`bhx8<)Nf;FXXk;~`6qw>f4>?zQw3Zk{k"
    "qhn7|b0d*Gb%dNL!e?o+s9%OEMJYOJ!MC2J_7W~*0Qm?jP<3^JQ>yLdF(<6nZ7tMDYg{YUtR30WHYv>Q!yt0sY0V@kceD1Pa;0nwsWOcIva!@*"
    "%0N6!<2f#nqzUUjfJQq46sHoUVz*re-rY%lcmXu$m}yILKZ?nf6<feZj@5o++yvEfv4_`Rp!6Vsk3w?juELj)jUCYjkqf<&?KLfrX4Ns^chNP9"
    "lvu4xp5ml*nnhOZuOuHqN<A^CJvfJymXKdbOe#8*V5KrTSFM>D_7SO|_ZaN~;Vfv6U1HWjU-=cQ?vO=OR_bnQ(v?|w#N!*GCT_a0Wpq1UIjQkl"
    "wOJBM0B8I$(~ug`gP@RJpkgSA&Dy&c#`oM9C!p-=r0GJ_Ne;!u8*<_Ga(jWZa%i}s8^pV@K_9OV8sF=Spk6_-Nc31`T$U`##446|8Tt$zh!S}#"
    "RS~LsqfI8h3h@;%#&V69uI}mJ@nRvIzz1g6G56+MUv{!ZAWX+rGEZ!Fdn#xEQnqAtvS?K@#aWkxu;*#lku+Tbv7%y=aSI%&{p-1}xSELu8aeCx"
    "QX&|XLu5lJ<*Is?qxpTuR60rkqAzl)VktOs5s|Q0)xjyR*U-l5C^E_f4UE8r&d@;DC9FP@T-k`!vO$HshN}dh)+xDp@l$~}a5{izs+tO*Aq@1("
    "GM);9dKS%nTv^Wb!FdfBVC%qL8YY^7OBPFYr_Ikkn1SOOT(7<Y^?u_wM_eLEvGGq8`_&F3%>*lcemEAHB1r&_*IOcbez9!MVwpkFpag)=F$nu)"
    "a1RoO!~27lhK$e<l{`N5Mwr~kt<C|jYz~^-eKmbSfZmffYGj-HI6Z6*?0d%+=?!qpJ*ZEQ5x4FSpZ(3#d$xU$^61&EN4}zPszv1b#ru%FS8r)u"
    ";X`ez3d0}{%y_CU<1;Nv-K&Ca8Yfi7^7`uS)~1=`vBiHueo?H{k?Mv+pw?H4Z#kOoLEj~O_>dlWr5d+5l<Ue+_UJ=Xi@*Ipgglfi"
)
_V12_SYOUYA_SCHEDULE = json.loads(
    zlib.decompress(base64.b85decode(_V12_SYOUYA_SCHEDULE_B85)).decode()
)

# v13 robust core: senkin13's modal trajectory (8/8 train replays),
# sourced from episode 89634061.  Opponent adaptation is applied only to
# compatible market-order sequencing, never by splicing movement routes.
_V13_SENKIN_SCHEDULE_B85 = (
    'c-rk<O>Z1oa{Mnm^Pv7BDc?9!uOuu-6ewzm^?(=*;9V?WtPf+~4F7j)huzg(m64H=`Cc`t8QmKE=z8Ck880I8r~f_s`!B!#^KZYN'
    '{o7AxpRPZDK6^Yr`^PW;^&kKJ^#@-+{_~e#|MPGE^Y!yjXYW7UZNL8Z=)(_R{`&Lv#}7YV-<+MFy}5ljJ74U7{CT^5`}K!EZf|Zs'
    'o}FJze*Ex$dwswA@#pQ$&EZE2`=b^5tH=L4KW_OCUq0Nt{rPMA@4x)kwxJ6@ojq(n-~ZOykGFUC@6R5`o%&a!KHc5E{qp91(yl}='
    'cmKATw(8S|H-G-}>FB?Xnzd`q`SGWxrV0&QdhMDv;O6?xcK6%S)2HKAYQ@U5;`96M_7iHyejpC*KWfb9+b_FE+h$FEBD89Kn7m0V'
    '{`9Ze8a;VDLCf(tY_E3<2Mx%0Jr46>TH_DJFjmJ6dUI{2W49^yAJpX=Y?%G`!xi~;*?hbob}wp<(27N^72_rQb?~q6w|CVxEb(#V'
    '?vtlE!80<PBZuYs^t;+vtm4&qco3Tq{a6F7SWa89>n_?O$770SW!GG3S|(53jV)Y^!?+(lJUzVfp#4BT{eJYs-T0%yc88yx3Fd9E'
    'A2T?^hXFnMq#neJcs#)<ZsX(a^ZniR!w=iLyT6{DUna(Kx`~w`oqV0p?Mq>VhgpRKj22R`LF~}l6N0`uoR^k9xWi-Y7!D7ngE{Wu'
    'S{TLY*0WDYZOgUr9stKJ-x<N~<EwSj75fu=7VGO&8ey<JP0x>?m58u%L6aUm@yHrPNc;gjNE~+Ao7<b4?VJ0b|Fpfk|8Vo+-$zR#'
    '^*&s+wh6qX{IDK3tvCk3V9fCB!&ipS99lOVM<6uFbsRr(a?Yw@b0!pidw2V(@Q7jgAGO$?17J%HoB#dw-Cw-Or<4^eGKcQ)^)_7S'
    '@MPBS!`FK+m{Ie>gK>X|@?K8fETwLmPys#KcHwBRDzmFcdaQScU^jXIn=6p5R{F5p#yIn_odCal`{7+}B~B49*nz_X!K1_8*+&e='
    '*6|+i{Sev@Foi=~$WwkBX4n%jP4;)_6emwRj848=EhCp*e0?4<43{0Z@BzZSq`n@$y;7iF_-)Jn0jC_;ikQ%h;i!%tK7WKNLo2_l'
    '(3QekJz`fdA6~4Lc=RFyeS8!oe?Zd!GDa)|NG}nG_~2Q98YDWTe0Vxrdq>0s5CHGIVkmq_rR)e!-&U0l(pY$SIEo`DFRWB7d}ul7'
    'AGqnMz6u$3Z<AE1Z_}5A0XuEfJbc?BX|+$HkE{mGAP)2v5AW>P^Y`vH`OHzKOv2BWNPV_obC{~xvW?+?hn$Z-;pwkxPnf?CUBYEO'
    'aJ7MBfSqS!)=@wj1g{FBbaD*!dxR<m`Nx;911si)VJx{a*)?bL*cFpyacw@Pb^=@QrbC}lU<S6+RS6=c%e4t#V+bCL`|M>{d~y}h'
    'VA9bpu`j&^e37B=(mAVhA@LKWgCpWZzgb8OR+khKXLu`i;Z%A-LR|(tK3}M-j~+ZlJ9hNj3vn4KgX56|1J`2TLg$W?k51t)#Jy7y'
    'u(=NDiQF!p5xISV5bX*B%;V5%{l|%kDKT;Oc()(bLhq--sB466k53uza}1$M@FqIDVs7~OE@g%J@dD-5>l`@06S%I8f_%ys0<4Fp'
    'AASU{OGlkP=1wm0fC%J57%%Obdh`S*CI^4$w$}}TeBzuB<a>)ZUZjkDi0{iycaIn&(aPWy4d?p;c2140!5(XW0!AV?rgpjW3@dXV'
    'X2W#~^@IG&=IaT{i2Hr^L+caTI>C4W#ZC-?$NlY(*Y~%Gwe4pBmg@#0TJeg+9BnYFW!a7Uqi*R9c&T*6jwb=+r2`9WP83creIi68'
    'B6|s#av!DHe1Y|9roh_1)K3^zTRC27Hftj@>|CdrOf^XET2hTU9E4LPC&CXkNoQjN4zj_3>X|q-wQL0J5B-Lu!om*G=4yv|-r*|{'
    '5odRtq2xL^RT5%$d8aN;$@tF8HqSu<F*FFydv)xA1Q8Pzr%dKWI?mv3%#QC6O$>9HB9gXVUVw-BFk}TOV%IdR6{RkN6oOp&fgWcm'
    'JqZPO1liMeh45|Hy8@jD+)iLup57&Z_r7)}4#+c-moHl(aH!puz|y%Sb4|Ev=JKE#69@4)cu}_o<N#ev#OM<BBq6sy-qN76yghxc'
    '|G|Jfm?lwitl<+bA2ZPy0Oi6Yq~Q*RoE4l2p#zUBgzzEt$R$8j<r7dF-7$nIafK@`bvEJna*?oH01RM{dr~rcU}|zokeUE~xWBiz'
    'H=w(8|A0Jc=;+jt!S*<Sqrj61Cj+D%8J`k!le@~CrIcYaX)9UI%-kv|htvDMaPZ2@+)A*x2_;9s_C;?qyZ#T5_Krq{9y*18JINXt'
    ';%s*Tf$)Kw)A&xY(iXxRsjFAOlP&2d`DC#46eGB%xPH|l#Oo6AU-;1X&GpB(J;HwmWlFY2=Q-FOX)ce~@j7d!SGtA}k^j;qMKRSt'
    '5syjD{MNST8!@dWc_%SLIzl<^cHC++nZWjf5h?y&G8oI)mVg8TGKt+kt6X`pJcjU$*mA06HVS{?oN_NF)SCLMQFFGlK&tSgs#w;J'
    '^6;`|kB`Yj9#R%$P6X=}u40k{)x#Wz0sJ82$&a8nQm9dAe{TiOc#lDn=u9+48CIV4S4*w4RSk3iqk$OP4OVx;n4#Q+1n$7muy7tE'
    'A8HfpybYd`HqfadW6?52t6IBUYCpaFs*Dqnzeue?JSW{Ba=`=sAhs*EUBNYa)7{x1TsdgSs+J9<Q^y`jWR!?4J0zEo7pq#GPZf$i'
    'hE~iPhxbtBeo9Ux_fX=8n?LWCRCMjvoLWN0GQRCmdPdONV`7q<>K7ALljrepL12%$){*@gZ8~oXfc3+V7%OiA8<0h}s5sS&lOuA~'
    '3h+YYOO#)Uiy{xu<^XmfF$KImAKU6IoEN|q*ccy}m{6+NIt0m&@YaszO^0bFEIv=Xaage|ErP4~5PWm-k3Jion8$T7IPOVI&;@`;'
    'y3q_x0BamHiaJ=?fG{VnS|k94;C09tr)DQi7sygdi-NJ1Xs~u|hi`Oi_!(b42TirZ4BiPEj)Y-3a8>8Nq|<E;1HGuh^zeYhxTx`T'
    'QQJ*O!p`%EV}VhQ$pUv%R@h#AOPO=M;HMtwk!xTT+$HaMK{;Lnd~URyHMFpzs-qYVJad^To0)zO7Yt^9w*_B6wqa1N6$!Q$9-nHt'
    'YZDNxM4P`rwPp8p=phs`VGe(`0ecoi;&wy9SgT;l;<6J!VV6|fMPPJ;-Rg?$huN^kQumAMZdPraSO9-$P&4pt#%83k=ZkLm5GPW5'
    'T;a4=M*dE{0bU8FlQ%Q!`vHHsc9O#m0!G+wz#e89xlSChDxn;PU^C0(9f~9P{mET0Vo5B@9+=l6!o0w|IPasd53KG>LD&y$qPXoE'
    '=-qyzhL(z<e}#<#mUH0R0?M02VJkHmfGyvouB)jd7sEV2Dmxx@&Gn>d;CV`5!?ke(69{Pio|!FkMggIszO*=JL?U17*fSY>Dwm0d'
    'UZDTSg!Dv&O0!-5hP7Qm+Z|TkscnGQrISZ==;UqB5WX0Jf>O!6`mG#Ct=-_Cmsu)X!7UzYnDb>OqIHs}W)04DN(G`v3VHb!@e#Dr'
    'C@jJfZ8Q3vj%RJWIDL^xzvC<SMd~f02sr}KDOuZ-8VtF$&IA08cpF#8wWHs>$QNnDJ(e2oarpD^n1_UU9RH9d#HB92>d%4+DLrXX'
    'v`(051O+mS3`8+e#~`nX6n<{iXjm=<gg7QLvQ!u#%xgdFV=Is(B!P!cBgs#}LRnFE>?3iAJ|*C)C?oqJF@>0FP}xX^8dr$c7*Ir{'
    ')F@aK(y4y-mAWr0-g|a$j8oPoxjSetE|=uzJJEQSlu9gLpq>Cx$yA?$R7@#pARBHD8JA=FtBSZM#P@hggg!CKO!2~%CApl+0C<tf'
    'WMU`izf^VZ)SKDtVOBTX+{^MAVE!ii;-x-G|F!4xd5)dCb-OhnM4{GhECL3*Ovas;qK~O2Glc~mAZ^S_+=F~Qp-3^l6AN>>M_eKJ'
    '-p*7t_NYt(2J)m=7lANM%$oL<ppXONsOqf4)nk&x5k&{gBaV5UGF6=@i!z5v#$g7ji-5&|wAKJLE8wr=BdlHXv;U&?OyZt4ux3>t'
    'sVGJ^FAl`nygU#Ge9>=9eA(XHs}sRFMbZin9Hmcl5k4ruG*ytNE$vEI;Yozlr-p=Z&$TbT5S&KZ0Ud_pKoCK-_nwQ$vMefpWVlDE'
    'bFL{N+GmH~x&|5uqHLN@ab?`;)i?+$jMn>Lmcp)XEOJOmMX9`-A!yvZka}A%KC-OZpd@^zG65I3<mQ|?RXQAuaG*8sh5*N+-s<)g'
    'YGWXPQc=Y8F+AD?Xdds~3V0MT+*x^4bk-5!9jmd`3r4iSi%9C@td*GEbdli&u?<L#`a|+$nppwGcI~o7kkfHQ&&^e{<O}5hWJe68'
    '*=V!G0M4d%GZ9T<k%Y6CmKMbzyH+MbLl9~62oNL|>npX~zEhwRtIWG~ozjXNC|1;qhot~ma-r|ETx{n{@YtO*<3J&c1U@s3929t_'
    'A8OHjoy@Z3lagqXZ7V`D{n50JW~&&2LARQb=Ai*~G1UP#%}yqIl`hS;v&s0VYGFbEmf<G~4N5LkoC$l?Vh!{$k~v9qvJhK_xB!Rk'
    '28hceY{gDX3qvS;of2XNrRqeRLX>^5bT-LC->-i&ExUeXj1j;9+kHdQUr2swpi+nRn0#b|N-Nm)zAJ<&7XKv-bUU_^UrC}}_I8gc'
    'D%@++2n2SQH}R`_EtQd_bLJ@W0D&Z2`E~(0N8+dtg5hnL6@*I313JoX_gUVJpB>b+Ey)FrQ5&p_LxmQ#EHGysv}&`fq$an#=VBg6'
    'hu>_D-J0gUr3u^!Xd^PYcs>{05fFfD^z!&;`|x%B*j7=yVg5Sy+AUdTI^#?oFi|egrd^>2u$C-I=d`w600XT7e;DrPb~h6jMFQS#'
    'GT?0#a8Rlht7W^z1>W&)vjt~;rAT#UaiLbyj6GIqx86%P%Vzp`&lF&z_3s8Xdi_U0jS3X3&FRpjlJ86bOzKg2lcg1UJg(jBlJb7>'
    '^&A0<5*Y-g#7H!qdYys9vH*j}#yWTAG?x&~6rKV1@$4F#u^7PcWDv1M480J50)QAG;<2lL{L(GFnbOe@YQwZ!39y(_r}+2u(dEl7'
    ')ejPviro=+0K;tidF7+6P@0s@KnHFlkphag2O6G-T$?~%P&KIGxuI)ALskguV5W#^e4Q-`kOB*B24+&9)A~B0+B|ecufLw9?7tEY'
    'F$Ra1-|u8%_8q{<A9GE_g9N|=nr7C{M5x-V=2kimb5*O@l;+Mbq6a7NWHfPOZOkHr7>ZnFI)ei=HLT(G`Sbe1z(2dkrm$Aj6Kk&<'
    '3A&0?cLf|*(sy0~?+fk8;1e;$A^+L9J(n3k5gma9Nu!E1@<P;DC4Et=%RJz)OxDXNqy{rUJ-We6BTO|j?<WBXR6r~9rZ}m<tYCTs'
    'S6a6CR98_M#?lI8d+rFNi_0E#AmOm%x9{GirJSlK62N{KnqtL4y~JjI9S~Q}<b9THpJx`GNo*)E$bpgJLG?7pvg;#*v5QF(8rf5$'
    'MuD=oVVcb;2s3o16pQ-lD^HCLf)pkS7wPV202jD=w&y^>1zQo!03;NB0)IwjFp73BCTG}XmbJLxpDZGzyH9ybYg$9;92O6p^fN>B'
    '46H;#RG3<%K2+pz5uHqP#iBS=j1PC?42rPg)+N#EH)QtEw8A?Ie+++)${obiU7l!;xE)WDjG#IZ*qw~L0oa3pE5Vp8GA1|f%PZBl'
    '$&xw|sgB65;AZ7#sBS^7Y(qBwpfAidhZe(%1-}3>2mJ5!U@|!@L85aFwd^A6TLl0Rr1XlROSW=>snx|}qWyNXEa#PZ3yJ*Acr~Q`'
    'geE+YF4jR6Y1-<$G|fkzfO}Oxl2N6|^m1OjCg^X0RrH*|O`*CplT+l#Lw(d}xL>R^5y+&;O>F^|p*O`3s$j8_CbxZO9}W>^1+j1%'
    'd(^Tu70v`pe<?i1Xj|02rUJJt>uj26=7FJ)iiL@+qspRDDoyKd^Ws&OZ=agV*TA7c!+eerMe~<&ib$HH#D{g1tk+l(=cc?>!Izr|'
    'CN-~f300T^Ij>Ts+E}S}(pve}B(nh#0Cf%u%<3DJOTFO_AGZEh^w}Ia31)akSdPdcGh%dyKq<3BV07!01Y#SI%QD`0h-d*Er>gbh'
    '!Gv)4A`1M`jsZxlbVO?4A9pJPe>NsAf<2vE)*P!3FmK5rZvuGa!rPdXE7yD!7&GhwVV&|zBwIk>ph}#yDdg4@AvvlKDU_j7-?mL)'
    'Nv<Z=a`}e`P<Y9rXAo(LGdN(8m`8Q-*b$v5*a7Kj^fv*v(CqD$=082)7M-nTZ`inV1TB^z68OUfg9CI*eh_<{4CQIvl_W63DvuMh'
    'pofCD6TSb>F&K!tOIA-grQMo}4V223I(4~A&hqT`T+bUfaX$c@mT0hA-KoL>Ac2iotXnCMMWp&g(%T}%fKfw#R!>O)N|K~8$Jc@&'
    'JEB4&a3_j;nagX$%FB+YHe;`OuBW}li@cOLH_%F*LW#O@OavGt)qas|u>e{djB4{m%~h#DuV`2gY5ah5qBMA2cWO?YbIyKG+gpie'
    '<79fP5Dg=baqB#G$*NJec4K>`*gi33s0B26|3s_~qAFFBJ@HM=uetalsRE(-;#s>lCaXkMroS`_mU@~q*YpG30G#EVHalR&wV^h8'
    '?3IJvR&xoLJpBm&hH=&oVGkwSIf<r8GGPu^F{1seX1jd~G;uYP{h$=nnst&u6zHwC%G5<!czqrifT5Nj%ZaU2HC;t^T$3}Q$dS1A'
    'Brb6)u*RY!Wn#oPDzFpo6-$Of#s`T<jr3}shIIa2W|4w`Z=xWZomI6b*|gVq$);frH*K|uy<WzEGI=UW=SAk^V)wO@nulix6@P!H'
    '##wZsJJ1KvfsaRbDq~^ga4*2ZcY=IUH#bg;K>iC<h8iczv5%$fkcoG9s?%e|LqZuBpj*;4hX9D?ZY+-15vKr48--iZrRJ!5GIN@n'
    'Pu;XPNWsdAZ_ENrJzP5MEe_OaTHP|yXUuMhu6UjWC#{Gl?N&pL>LxqkSTJ~ON3%|G+>odCz8@Y8DDM7Yf=OpaW-!W3Y!zdieC0&m'
    '`?+*h%jc6>ty2)j`a5yxn<5T9FB>RKrIj*aFHw30s4@hFd6{$|h)yR+1WF~=Htvr`!_s&dGLUo}mwkQi-7Rt`Da)@QSsz|#12{=m'
    '!sW<2(v7o2*U6pl$WFYIMUh1HNeMz&4B7?$l2|>^lZ+W#IY6qL>cH4!zwnhh5}NxAx7wjNg@jWkHBcj0)mJ5tXH`HcdsH+gkpKeF'
    '-JsD4B;Mqr-+F7W;H@MW(FYTNe;o*9Xqpzi_T(CW8vRO>e2e05umLu;yOS8=)k-&c2ShVrHDRDF)(uBcg`ZBQN?!+AH8~eYH@-r3'
    'CM}<3#qqQX6b+eqJb*FUN02g3Lm^`r=nRdn6TfhlDWDH^39LiVMmpG1?Y`qyUp=NOaY`;o%&<DJE+cHLvLGg+2(=?Lfg-U4(sV&&'
    'NIhbir$INv$^~9mbCQE9C{3^8QoMTGOkAMu{y2FdL|??Fc6St=jd_Ed9_%gVfkXTa?ernb+1wosg=Ta^-hh^$a1lAJNe(mrwKKt_'
    'ajuta5ug!Vy^)Qz4!PiOlGl+s`os)WtIkd@%_Rem*}Se*1~yJToe2o$y${2Rl|rIEOw#Rn7wB8h<N|;-k_>HQG%y42VW!1Z&RJQr'
    'GRo8`_c4al4ZbJ+3PxTP9a5cd&T=@c*G>>gWlhKHJfM9O{EZ_}0i;DqUUub&8Q*1=Z6`v`5x5fU;Tmyp%C@;G>eMBjn_i4MhbX*$'
    'tcfQnBuCqvx5A2mI!5L|)H&K6ag?PNKPCE?0%PuYlUzI|fHTxwrFd7Iz(`b~OLJYY$%xbFVAI*j(75P@R70t%XOnTQem63%#fz13'
    'X+C!I!)qtkY>6sXJ)mmopJ|;1E`-0;OQGU|UA6pmGmDTb;U>J5$|udMzGKJ-!>({{+uOrA8dasFR;m0{-|||kOp#rSP6e?Ux6W|u'
    '?uN+#DyO4d9E;g2tFuKnM0=$znS4)lnnU1PxL;BF7fz4QYZs_eayFlkHi{rj6167A(CkbFe}s=L4?3A1LacO1f=v{N(t>a=I@g#Z'
    'O3o*SOlDj#N`?_pefzN!^rl4n3t&RWM#xzbVxS2#NWJmC=uIwJMeT-aZn&W>44GUe?PYLQ{)S&GzQ>7`Nx$|<H8Jh94p2g<%O$iP'
    'ZB+yVK<tQNM^+j1!-0@N;6G6e3D;Lj!=Db3Rna{=iaw+J9UhrYpptgKwAdY+Ro0+rPS2uK&m($T&dsM`;?O#!CJ)LujOW#}@M;8V'
    '>kF+6&Pg>XbOwx(`L~w@+@z33@%l_ud^<Red3E+My%t$UET&|&t8!UR7iQ7;um_P1HE}E<%*nTwv_v>;;KlK%M{0@cj9!BHGch0|'
    'l)i~?x7|M2zI(Sj7e|p!(Gw|P8nLQv4+&qW->eS%l(C}vKszB7t8h`W+&~mrbtp$hp;-<m!`%~+2AcPe?%ft-0$Qc5IC6zs1OfnH'
    'TQllfY^BQ0pC{Mq3!w8#T}6j!S={uL1~*$zSu0T#jAOuq*}JWVEksgH3>32HfvTugWIvPH;(?_O&mM_v>nH09lG<1!D)5-9u-3UZ'
    'am}xq5`q+%nbv9&<;S`g4<nPrs!k95ijDRZ9EmhL^2?&5`QR(%Bt=xk8Z4y6(U{y8K+m-p7^02EXy@9Dq=i!>#9{45J<sNRvzc8~'
    'HG|EKtg71d=jsGSxDT|o7gjUtK=V0WA#xvD5&V}GWa#vcIhpH7-S3Cpj6auu4nwMlC`&J}rFiF*Kp+VDIPGHI(isz&9ARIp(w?wm'
    '7RBh>(*Qy=7iW9~Ak0DXvmB796=HYpb*amYx1<Csp+b?P%Myphc0kkdMU)kF-K3Z|!O7|6;NckOvC_^~-ZVFs{bnc%pd6>fZHa3M'
    'MUq84&dG8aig5%}xN12aBMZ|U)Nv7!33scuEe#irmsUp>(_O5{ZToRyRC4RiWpkDk!Sg?{T-=jOK><ig!05oaq<yZGT~E0k++jqk'
    'mJX&z5*two5fe3vE<7>)1L%5KNq48^fX8G@U|l{YT;w0NDr`a-PX_0is+Wo;_kr}Am2{Jet;Y&AuNLh(vy3P_9mq@l!W4F}38&K|'
    '#O(m9r7Ofc2{84%rN7qwHDf}0-rMu%f|#w(s3ZfSOu^^J{>uw+V-BTcRi2Md0nu=pR9Dgcjg+6I)^rQ$%*L;;bF*o|7Fw#^U`txT'
    'vXlu8D<M@X1r`2#RD5C#1q#!$F1fcb?;m-<*<HUQMN(d7vcWDZgeT!0_|B{k(4m)Q5>^Ver3sR4itBVTr=j_^R{4kC#0M0BYP~pj'
    '-+7vdhzFBYW%jMK9IA9Ul(bD@B{f$d_ZrIaGNrlP1LbP5td!oG7c1wSJIWm3MO~;nMK=sH4hzexGn|h_wo7R&Je4e1zVOXx=dScx'
    'R?3W*=6xyDhY64)K$4{k+hSl+N$GGd+v>tGax)-Ig*2(y$DHE(wR3615@7|5ruDKCE}}ZRPFH)4k((5^NoCBbgS0opO+8xIImVeA'
    'jo1M`qBkm~!3^LiW@wk0&e92@HvK%(B)Yyj-(4<pz)abLJ1&agsjq{vX#~FaDH?JwmNjq*Z;Z=sTr&{w;6^UKMfzFZz&jQ?pG12Y'
    '{7=a?>N)JadBL@acBP6zq?O$Y@#%u<6kDuQqMwEH7_Jwph(ceAL}`>w-A@<CA4-wF2=i!lmlm=_Rv#X`z$11XgTw;TOcl}tOGv3$'
    'CX!?lXr2@BUu9uF;8M>x9gJ`WzN*3#y-3x>&DhQ*uyZw5s}Y7AKPyF0p@}BZC+OW95VR&OUIu89>nH1XEYvjSd~TZ~hD_UeskI=U'
    'kLZ7r$qotYg=-7W)M9URjp^a!p~XDK(hpW~-MQ#^IAT5RgofZ(;=Mp2q-nU7VXQ!{<O(mX7A`t1&)-c|&SIZT50U6;b{11^Py_h1'
    'j*)U!L=HJ=Gf$#2H@BxVVE`7RKxZuBjld6!tf!TV8XSI%Eo=n1SiPqLGiS~gNXQBxaczDY8ZOk@uwANQ+u9ujbs`Ul3bYWPE7J_s'
    'neB;bxO$h~UELFy;jioOK3!!R+*K=3S(fF_;M^toJG>-voQR?^mTt%_%jog@cvI;>A9WsfPUXtGp^Ik~z|I7CllTx3N?C_`c*D2v'
    'Us#kif@1z8bYJKHFr^TAO9KQg9OPW3DJD_0A8-kE?&KRy9>iSx$opjNG;s|}VYMw4%!G?aI1dj~JDp@riU9rYT=!1WR}$D&DqP)3'
    'RSEV+c%i<V1J&^uW$eWw#r1m!=CQ1T3vfl2uJuw7u{}E@f=af$duRx(wNC_cBg-r*&*N3_&c}X*o02FD$mB)&B#VDUl~N*0Bh^oX'
    'l|ryWr_4$a6$yH`iq|^>_(b=rCcPo0(MR<wRq3ZJj&tkR=R!`M*+VMfLh<l!RV8$W7xh7<jAct}6)9Bw776%VdG-4eymCtY;x6fu'
    'kMQv4jB@cbAFulgAFo_0)N=|SH~oc=^B2qOg7=(C^*9F;YW1zF9vg^JsqP8yZAsV#qg~xYSB80#a3iHYleC^ib(+0a5n=H7RyBeY'
    'JNzQTo658Fqw1$y!cDH;_*QX0vh$QMFlW(eXR9F@Jh6YsZ|42|SJZ&LN-#kfLxOqBIcCS2!Zo=Z9p4y23-MgxN#!4gSQfO$go|{Q'
    'Naxa?z3EDyO;Fg=v>!DbNT>)#p!zMNl}5!^yKz3Z@w-0s>@s;P-1!T#&!Ql*r4NXWf|d$Mt<Wq~2&z2bAcG8F9+7*zER9)CF(iSc'
    '^+0O7@sXyL1ib<oH4Cl{r=(HYUBv3*bIIY4tp8q4=LFQF;-xo0P9pjv&B7%~xKya}xN~*!sHQiQ%T4F3=bs>nWGN=PM7ifHyF6iZ'
    'adqiZiF10oJh48le9Pu@iU391?y3Ln8;t5g0TL#k0DyX!?ITC}f=ZgF;!?Q?+RNlptu?aIE)267%1f_<c#$)Me=$6&idgNHG=<A&'
    '=2THXB9T+|>`W@ca%pm`U(Z6IqQo7w)LhRZe+hRmfMtZGsaibUR+Tz`d1MI-bz4hB(jwPNIcrBav`uPqYjMajQPYnBsU^0QyMOy|'
    'zEYlqRPIH8*;rLEMGy?LdJYq$*~8<-a+xssGjb4`HxR1D!?qRa#b`W1f~e13fXF$v+M3QUK(oO9wFRV^FP>IBj`0|Dk3|u2skNV<'
    'B2>TUV-2}OUO3+*FoVsBqSM<`5<JhSlB2f9_ZrqZyEP2BBD?yyWg~4yakT~8xHQXx9SnP>v?l}hgOe=+i!!2j+XqJJidQ!ykXdi>'
    '4D2QGvnK1~DpTYSw0KybHas6%5skLgT)Oo@P!1YB%-e8H7~Idnu-04p0*uoWnYm(yl+DZ>-J(~{qvq$;u$QXjl1{yVmM5S*HNq1?'
    ';l4mlp+ht4p4|)c2X49(5QBA!d7<efN8>6IxrRJrftGnox-F3IH}3`ueT<K2wxBPIdR50$csg0Ai<MoUXXpv@h?0XVB^A1f(pZ!g'
    'f;?c1<tj0y+6pwf)v6ha#drcsnL0mBawg;14{j2IgIYLfnc@kQC&+ezt5e7j-R;JsGfIo6Dek;f6d_+OM#K!nzlzPtEqbN)BenTf'
    'wav#RSTt^F(+s<tqco6dVUlK!FLU%L35mYk1=CFKyQFKHy?PK%DL+x<({!W>Bg@zsG=S1CF&#THePp>X6N!B>=-ZxvD%544VJdls'
    '@#8_#M|oGO0Qwj%2*7(O$-1h^q}K+E4)2s=Cd#E3LIGmoY8enEVDcJRbFuV!-2Cvt6N=|W2C3I>6Vw-`J`nf~6W1nE3IwE%5bfO3'
    'thxl-s!e)yGWDAHy-TCMR&$vgjvt6wv{e%2=m*4CwOgbqyVxAqPv<iYMWLbV;U#VLtKn%l_A<N>uyy+$39?XoMW9)lJ8=K*TaJ+~'
    ')LU?aX4KXs#vrT5vf1&Lu{R=iBz&4J!~M(%0I&(%$LvRh+}T(fYuI@(tK?=SxTl)(B!uNSWeX_7zW=AA3wlf64V(^a*<V^cI?6lc'
    'E2l}DGmiF%L(VVYl>z6LP2phxh~RM}<a;o7IC>xA$nF(g2d-CfZ3!fd?t*xDvQQ}RWP0Y@T)){`^M>Pd0RIK~MX^E(+1F$nXNmRj'
    '#6p2et`{tTNFDjcQYt_GzQ|445Y$XJTIz?JcK@Z6?-1DI@)gzew*K+|0jwJtdj'
)
_V13_SENKIN_SCHEDULE = json.loads(
    zlib.decompress(base64.b85decode(_V13_SENKIN_SCHEDULE_B85)).decode()
)

# v16 position-diverse route ensemble distilled from the current top meta.
# Full routes are selected from train-modal traces and never switched mid-game.
# P0: episode 89722372, modal 7/7, SHA-256 4e5f45099e4ba290d1d7ebf8ee392821641ff2df5e848c24bdd36b024f0d0855.
_V16_P0_SCHEDULE_B85 = (
    'c-rk<O>Z1oa{Mnm^T7TfMf%2(dS`^?N&+Rhu^teE0d@@o#`-Y!&G3J>H0-Xfs*H?`%=e0<W_4>6n_chwWkyCu{`|jZ|NiT5|M>fF'
    'XaDl^*@xSYAI~1n&;H}r|N7g1e|hlb<3E1=?LYtiKVLroeD-m7cXxJvw*P5&`|0d_^>99awE6nzr?>aJv&n}aKL7QX+jnpOa{KkG'
    'uix(v|NHgv$KCGDmxq7Z-MxQzc78Q`eE8X`hll?=A2;)d&u{PE{PLx-pFaQoilMzfpY3-aKmGpFckk~%{dD#)?LneRhj(8dAAh)i'
    '|K{`SPoz~J-oF0x=MSUb8#U|Dn)C6e$EHpjNZMr@ur>|2yM4Vo{NLp5^YKcx;^R-d-G^K|jsr1#VceLH?>`@2ZJRZFLMG3&Vl!{W'
    'pT5rH<jvCwT2IGecY9bkXh6c1?~l&P9i7v_j@tP8*38iG=<$P{s(5K0e+;h3FU#iLr(qLL-~D(X2(MfTubi&WFC%^XX?HKY&>8CK'
    '*-_Xhj}w50e>g`@%QexoczEpU9KDE42t3x*R+y1EbQkTB)854sb!aYY$Xaj6VYuP?9HwIU&@vNfHM~3hebR=*B%v8`=ZFu;yxWI?'
    'h0EYH>Z4E2X&oPjc7*M?&OVG^qYv7Amn&m8`?*s8>`)HV2S@VpQ*OG<7~1oK-XpLp+r)}IZ2>ce!;9%)&JDJB6i<s{7`u4`ozc{K'
    '3oZ(9`0_0f%o|^=v#!`D_GH%A=~fwGusluA*=8j6mSaM3TWTBMR@}}8_rUDcDU<PJ0lmMw+r9qu%b#}lpWfcR{kO@I@EW6?AG`(N'
    'HllZOv3o&3TVFJ?Dd`KKt(of*{{7kED0PY2u5a$&e-M_yVLgsoV-E?i#!m;{qhD+f!%mj*%(CHEaoxlO9g>dn*$*GjCf<R(&0>0h'
    'USzv4+M8k1i4`QZyfVa@_8m3|b_;9+p%E^+p);rlHvRN+@TTwwPZmj^-=hr_u1TSRr&m1w?Rndmo%_lU4r5U0upPg|ugOLWJiJ4T'
    'qH40de}DJY0CIs$6?ND?rHRArE!eu7{^n2Sq+dTQY9L7-Y+p;MhZ(5Ne9ljwy>S&G0Gg+qv1np2tM#V!fs*Fdr!bNph8ghyn{Bg`'
    '^=*f#!IGd>+VD<e1u2(=HP=p~0(vt?^Hb{y{j%K_;SXC2Zuq&AvtozG{;q%VRSi5~?kFgC2Q#YI@1i|X_~Bt9e8ZHZ%;PB$2L9x2'
    '<=*#_et5Q0rJrpv@j#Gm+u!P|0%A)VbG#|5C$rqZ;D0yfdI4Yy8gt;9>1NDXOrw(x=b|tD+WcfD9F1Ola~7PBV+B#P@o9ejkg)8)'
    'TIgiixj1eSoRcgcWZEri!gWD%q>g+y;l7)2-%U6dk2`>bTRRSGA7Fe^i!T*3UEa{^7gMd+e2JiB$@rtwupFlM)9wAgMxQ&OWa#LM'
    ';o&KyzQIIXf#2zMZr}lFbdu0g@i`+1<Q(39dNef79gB~j?r--$?C$UXI>LOW10t}KkOOdR>gWv)$qszcZLfP&`OF)%2dq__TcnJ>'
    'H2$U0LfC!uXg}e~;AXS)d-FaeI!73M<hAw*7>V4N+U3q!i<C7Q8rc*~1|CrXqNS*^HJkdpp{)}vxZ9@_#G1FX8(_VP#`pT6$vbf$'
    'b@NklU90fRG}wRkg4I)*(rP}+^MItdsbSK}=A{_bn}v*Od!av}No_NG3L~>Bj?A!^oo2EGq}DMyorm*eFofdub9d}}$2)*>+|u;o'
    '6+KbTIb0kXRk>(<sYBdz@UxUUb?3DHK(be-4h>7phucy<Copw^4V~?V&Dse~-f@=0B#En;zi0<>@Z5%ZY9!=I&^|Rh8bE#)vS2UI'
    'bPyQtBq{~^_Y^b;+JL2bkx&2oU94gIWcFcErswpOxOz4W`Bly$fw`0t+9<auYdy9Ale*=$ts2*nQ4222R?7*%a|uH?eml!-9h)5}'
    '@mtOUs^0;i`+L)*42zu_JOHo?jr6`5&*Ws_5quq|W4FAIK=<J#Fbo($@}U=5usyZTk~~=*_XJE$k1A0bO_r-kX*M%}jHDV|jV?F}'
    'i_wh*ToV^Gixo%WHxnKLZ~}CzH6;u{c_OohCG|W~k~2cuPi-_-xiw%R2cJq%h5m({u1<Xg$0&QbXS1gKP3fH;jiQO_1Kw;YF3C4*'
    'l#EqZq<YgrZ0KXQ-`&1@v#brzz&*uWxxj;j^32A8;B8i-YSkB!7eVj>srj<%Ft{p7%!BFaHN@<aBs|~jRQP0OI7-31HTxA&|BHW@'
    '^mj5!0>BM`gje@#73+q1BZOST7EjGTQG)j8lyETPlhkiQO$y3Fm%^iJexV)Z;cjNHj~O(I-A^!8AyknZmHAb5(<F7rVmS;`2Tn|W'
    'P`y2e8ils}PGCgyp)9-XEE^Y?zQba)LPe>irx=GCz#!2nT2!P~nS6rSNl+yq!r@`b2gzpa6gjW5b3w^M4EFCiX<EC`(lH+bO_cHx'
    'MshxK4HQMhUQ$Dai0!pKGCH{<RZp|K%c(n`k^i<XXbJ01DP)zZxt`7AD0-H|RaJ>Z$<F36{o?K2pASoParYCFdvT;Et0fmu+aOR-'
    'bY_ir$Sn$iY$Yka^~pCu5iHy&0NoFd7>R8LEs#aGTKC#Ic5+0nR{?@RUu;{6$<Bf%8CD;o(GiM$i^6$EOm`B~b+L;Xb%`xH;0y&<'
    'v@%KheK_OBc}WeY<;mP6xMWW{y*w#KxBa~&7vN9Xbrq9W0Va}eGy@#Kf(DJEN<r56%OxGH5->z?_c8jW*$MLnvX)$+V5}uhsa@OQ'
    '^?f2S*i*{em_bwRFoO#~gPJgMcX~9ZespMSXw^jx=7)!IOQV+46>WDQF)+^~CItpKW~<u`hbrwAaa%fsG*~;&ldQ6Q;7g(fs^NhZ'
    '3Mxfl;3>Kz{<4`<5BaMtLE2bG?yyg0hf;1ffvpkk-Oj96^b(40FnKir(Gmg}5QS*PG>)h1t6ET4h$*SDzBCB1>cT$7B}b}G|2S~5'
    'Fsd;k8&Q@3d}yF7@M{K~&xhNWfpIvq*(2|!)gP9L-3Qag+Y@#2gooTZ{@YUR225V&A?sw1#!>*l`5U9#*vNkK_D2VisuC_VY$Lxz'
    'RuKjT=Eb>k!alfQS}iC2G&6_=PALKV`9uvZ6%F(XG7pw>AUI+}_@Nz|m+ji5BFsRHg!=D@GJo6)(?S1zDmsreH|nhoHf@m(L26<n'
    'l_;1-<DjW<6v|*RvXhp4^F+`zm1rI)5kHPN!1|cf2BoelzhG@&&~De%q}6Dfbkr!aj$v#09Z*!lrgSjd>f!hxKxdG)+s<quL@spF'
    '6twXf4?y$aKpj_bj#IV{Jw(tVPA&_EcA!;9K@R3g3n7QN$OOZX>ELk&BFjW*6C6@<8v?*6S;CGh<9G#r1MH0`;A*4TB)}Z@^OE`M'
    '-Ovq>@RkM%)-M<JSgl_=>@j}GeC6Iy)kROFLx}MzU5X~W5iiIrJYerc9fN!$QV+Gi#0^A0h7S=lWzShUi^C#2fzKdiaOjMW{1mK|'
    '6=lah%Kp%&1cZe$NFcE|L^p)yg=AQ41$l9XA(0$tWLFBVRzFcjU2PQaJ-aui{%liLI%qEt8O1ZtjqAiwr$I!(t^_*+3!9>UnY&m;'
    'f0G)%;pUJbJC<No$=(T}KkhK0PmJ(VjJGn+m~#;Tg)!-T92A(ZQp&<+uevpPVfd8~HG|iN)}AYP)m&X#h-MP?F;H6GmpqgEJv*ah'
    'NUlvr0iT1`X`IAoOm2GMwkPH>2H1%h1K-26wKf%E++!b}veWdKfnLf!P0Kk@$N@1%HSFPrz`dJ?Bv&J{0+{?7Q|n~8aAKlTbn-C;'
    '7!b-q$^g<*tCe}4o^9)9F3Pd0?e#*rHA$@I_`{c_=4@UPPV=o`GT?`PU&`wSRBF3@5}a)batb#NrB7^RE+3X*t{^d{S+00mZ?02G'
    't0Gq<1bJ@#bO}K&q#ed#=naGpRJrQKm@OB@+#DI@;j)sM^o1D7uO=GtaBYy#Kys4ibFWOuCKW~p*2nP3s-v*bk3|wG|0v~$GenJR'
    '4pMy!Mo*S=Feo*f<&1#XFA1|Vfk}t3A**?n)ENS}i|UixV=Du3GB?zo>5SLltQ`Tt$9uYhF-1IhR;`m!LO^yMD=^gyZM3+JNDJgN'
    '7-|NcBrsKK5!;tk^**Furnwq0qpm$S5%`20_;X!d4$ElaarL}pM-Qd;XtT%g(Wa~_mV%R5JmIXSvyk*zD%EPx;7ZzRk#EyP8)Pl#'
    'MJ{BbuPd-t<Xy3vSlsFbFqR9n*H(L5J=ab|9adz$fD>%H;5>Rrn)%*sUAbv?VpL2gfdf4=6Gv~dU0;GVNI`-usx#htSIy!jdL1iG'
    '-!r2&)4DVXCB(+i6grh$#CM6>4D>aU9!Ye%5OIcz8VNAmBe2F!OAFH|6v){mgt#aNDB2~V_)Y8p$FT`52z%&9MlS)3xZOy^VTP<4'
    ')v~l5K+V+v5}M7y*aU9!4UITlR&o)5ZD<Eu(ltpmb-fj0jx5*1jqyfUw7UhUnA)tdrv(;xP!5&L<{TvA5q6n5=jiO{Ij<`3hE7G8'
    '*?EF!85Mzf`S{(<2@YdG2<Pyj@`swon3Epbv`Jf1(Olkh(Pt!MeOSd@&@_QRfz*wRlOFPIQ2M~^*;Pv3EmxJgVXl&$hSdXWJ#wZG'
    's4$mj)Go;r@LQ-C2nJWf3^na*G(C@_p`^tszz|zj2*@IJVt2R>I+Mw*1yj{^!!yN_Q!zce=^NqPd@&R5-)D&l%jx3UVSO@K={u_Q'
    'oTw5G@t-d~jV)+XN#CZKLbUFsYRnu8-4fR##d`5S5bh(`Qo?tjoE(WKR8NCXtQ0VKY^-y~P^Q4>#D!X4=wxjI?0}Fzi<C3|1Q;>y'
    '3yuZswRyDg4S_iT6awreyGN!MIKR?F)F}XMP#fmmN+8gb;KhHZ(mPp$lz=pp*~KEz6i}-Q*6_MZJE7nyn*oJb_qfNmI<QGH7ex&R'
    '4JSnUQ^3zCSAL+*r7NjJ)(q=l!=eXqaw&}}(CcQ1D3xHXuM?U|2>zJ3)+P9$Ptmpzq&upl*;|2=O_rL7y#(0;7HL+SMW_WW5*cIz'
    'U(8(15_1?`L|1h}X*-EDJ_av}glaln1BNv$Saq_(zHBp?j@?jm{445-tn=F{L3dpTe#rI;J`wpI@}D9pt6>H?6O=%1G)a!!ZJ`8k'
    '2g*=q9En%h;1)lEMJ}8YTuT#k_IU-I<uI;I2bd=T9#jx7b6se<Sksvu$1Zn2{#cc;T8x8boo|nim8<X7K|ctic!eARu6Lv-(Z$?='
    ')uvZ|s4DsdP$mY_*mNg~m(AC5stbPM5;tYb`z(uK_szIlmxx<n*aKz9Q({-sT5K=ZOj3I6@{)3?>`}6qv^271biNjI9O#9athSj@'
    '&4_?NdK@s+3`D-|!N4WKC!yh2x<kam1KfrNU!yrbO1@&H&<b_-`6r7h>2h0SvCZJWbn!LF>(zrw98VB3>B#s;Y&A{si-J}$yqqE='
    'vr|#TAa^dYR}aJ46Vql3Qut)}I;!#z(}}q&YqJUMQl6ir@y=w#62MlBD?O_VE0HQWs9MDwQbgY)$zV3iiXW6^32V3~+1P`QXenX-'
    'Mkmb!SUXdImPV3g^AZ5N*vh>x+6BNClzA3|pk&D^ss;w6JB^Gam88wTc8UOcWVyC-Ohvr-o>G?rhC~kQ!g5PYuB{(OikIkET2Bc!'
    'F6eK86!o0wPoY*e%LB;~ofZUkXkjPdd6Vd{0_<d>(|a`^v*K#E{bnWgjtSlfnKoV)q*nVgoC)R~Q#g%L%&4MI1)gc(^h7fcG>%m4'
    'OyoXQ7L8I$TNmUPV{TvoVzbgNZ*XW(2Z4i1(fnoVxFm&P;=?-f`7?go4KP%6luJ=FC)uEyLj1mukTrE1l(xySmxQ-mDjWwiy1-El'
    'AMSVluJidL3>dQhe0UsKR?5k%Vg!pIvFCu<=#no9>^9)8Wh4U;wF3xiMHS>-*wao37ipqUA?+9d(bNp?SI)6BOCiWj4FemwbQAd6'
    'EMaK|?c{a@2mAvpU6K}@VJbNzI3^IwWi|x@P_Atjy#sqi;KE9DIX25?xn8svSU=?(+9xj#dRf1C=;bj3<oYRL7GC@5W$7FBfLVBd'
    'xzLgBL4OlC3w7L@B#oW{Y0;^2Hb%$OnP!y0TEalMVz6ya$rECip8$Mewb_Z*(Y_!}Rm5ai(wLl-<&?7%D=#!B{Dfu7OG^7Js@@fe'
    '0r*&=!D>aTimy#f5V3l>QWA`)3B;_oiG^_E2GC^dfYN1Dd!ALH5&)4-sb!LUOH}ucwOi+MQ1hQA+d0DFjOu2tv4wIac>#7V@s&EA'
    '5;gQ#hGF0o`?bZzVsLGE=@I17RGkXEi<aCq=J-=t8_J;4P;^~%I&r4a5~HHtlXwQs#)o>|+hj2;P@z*-0A%r1M%zean2P%1BHS2&'
    'SvlltB>ZgPRcdFe35{Cv307vRxJ#m4F!oxmo8TBdhbryRUl!|7Ri$i~%aUp@k^`1t8|H$HtY?8-!b0zp;+XUo#;baR!3@RQd6G;g'
    'NiaHGvWWPv3j=HSeoKOr{h*W*TXd3uwdbw2%G5<(czqrifT0#&%Vk@sId($AbEkYJWT+7_phT#)0&CRrn;4UgX4nbwzmiUokyqlj'
    'APvA%b8fm50Pm{Od`M$Q1(bHNkL)#75__mDm?$+2V~R3FN)1;<@h7aduxSF7Sm`k;WkQPT5lDr4T`8>=Chu-m&%ho)HvyiAd76tD'
    'Up=t1FC9yNrB4>{m4F)u5)d`26VCJF<Pf*l$;y26G7*8y47N#us?&nttX*jna1FcCV28yPB%FmPU6Jw^!6J!6UAX{}T`BCQ28n1|'
    'LA4e^veiS4jB^AxSZ<l9ccbXoc~KwAZZb()Cn#}fwx^IU?%!d~%4I&maI+go5eEw4N7Q?oKimozjqmqs=bYU<1&1dj&=qrz{x}lT'
    '-+4XD^Lj$}St(`q67#13>4hLKFO%2<;pYT#Kq)2LM&i*3RT}qVUoKTv&X<Cedl;AoaFLw4mLum#kz<FBlRL|iR(VZ>XjP)}q!Kuo'
    '{kf2gEMV<GB{HUE<p8CwqeJUw2q`?GnhY1yp$LS8Lw2Tr1+X^N2#EkZ#Z#O}SzhP{3!&`5W^q#={ncAmnlMr*X&J+*!93tU2T3i|'
    'Sqt^;ZpsA)H5!$cquskK#n?`W&|9qtly?$vt+2Ss2{mo8XgPu-eE*UFYiox+xULuP3;c9)fKEccLM189LtZK8D%afErF3*9pT7wv'
    'Bmf)dr;t%?<+RZe1nbo;0~W&j1kRzAp?SeiNdSonUk4IqL<(oT!LAmd|4c`v3z7poW(wxKdQRkq)K!)_9M8=NsvtLoqE|0R;<&r;'
    'O+F!_FN#wOCq-vwUiqenhBp(pA^L}Q`jGZ)E{;Rt8eN$;py4OvL7rCqhN%GCnP7=;uBvR&J{L5<k;1hO&EQ{>*O5B<L@ZP*AWviD'
    'k}yXff)!%Ph%9XEv^o<I%zGcE6|1X61(u{3yH*-9EYi2CJBI-_8)XgcVjxi{?Xg-nNksFsAu+=yPJKxBNZ2Ph87!CxH9e80`Q}86'
    '!&>qL+f-J)yc1|?Z-V)8h%3O%Xe{3D1~Rk}9wtK75m*y!V9i-9=6H%xEh*bgucn<UB%~<V2I%Nc!Dqx2XjS|tMH@jBI@%?1pruwn'
    '-d9$*dt<2<HyO-~2{kjsBz9*Mv456}Mn{hhcZSqe^b;9o>2W%d($vipWpsf9=%xhgY`$lD^~v<oH17U@nO<T>t8QGi^v|@;!W_bP'
    '^(w5m_?N00F-tyeEoUz#ge7E#cTyduWwZDga>B4L>>KwMcaBC?sm)cYV^yxx<&yce<dRLMHn7FP0)vl>a=7J!bFi4`?Xr4ZbU#Fr'
    '(4f9y8L}Yxo`Zc`h{Y%!)Lfz&OK8(lVqNzC6KRtoiX-o<iTk~FIo!JHG&`5Um+-OY;WhJvEM?nsxJ#}$7#)R7ekCVGL*_cJVI?D%'
    'kNd7Rk}@IG(Is-`TeMB52h*zr2hPu*lgnjM=^~l`mH_}fq>fE4%PMBjSe1dx0XRkiWdY4uB{Sh9S5kioJxE&uL2q>_Vup}7kTNj-'
    'XGu#!z>Bm{G`!YD)$Pa*O*4DkN@J`z(RDLysLEtVuuwu9x~ZBKD#vv7gc({Bb7NbAfcnTUvjo>aI=M&S;l5N5h*Nt~s1Fz<^KUN+'
    '5K1AKVu(&kMPCr@5s6LA%4Bu0a&(|BWTxpg01O_g6Ip_(lR~1>df-rlXW6430pGhh--Ad)G4LbIz?1H>Ql3FTU1Ig!_6fnM-_!T|'
    '$Sl(&L_6FSVR11px>J85XoX_397l$GfYS(i2o=lXZ-GOg)!w2NSr5TrWp6-FAM92}f^%JAH@=-?_ArMmbj{*srm}hC3bi?Jmqyxy'
    'p$ynENA-|`!yAgzQb!7}YYDt(^gL}~sXVktLEHI$yaI{FmDPaS)0OJPi*>&OA4;dsb7|zpq@vuw^8d@`OY6cxjK>n^s(MISto+9h'
    'aMIvNp^J{=-p|!z$yy2Nv5<F1Epl6jU20!>!!;@ItX^iBtjm9T?VO+#_leckUpohy3Fj}oB)N}*2*S+@i1YNaJDFgjE*iu_;m;+)'
    '!$2jH#iiHVQUrKP5Do-voOUr$>WsQfj=!(fYftDni>d0nI-Ij9Ror0(?^(rjT1C8L`7dBcIO_+=*LX`x&?qX@Jld29Dz*b!;$Or^'
    'QP)k1Rui0@p8Fk+aUSXHtN~8_Us>&jditE~HRPx1!cIILQRGs@Bd=2{67}3ynY>xeL1Uz1>QK7UBvrLAw#(sK@;u^5mAZ@2xotl#'
    'oodf{DjdXU5*e2+L4Sm3Q359i8Yt~Eqip-t%H;Xvs;hH76RqMpm@p|*iR-xR6kJynJdB0I2J}EG3!iLxXp699Lk2eut)8bq5wsDM'
    'B4Z8ax#O$~T&L!)-4a)c6+<EEAo0|%aAEsn8JZgFtyqWs+7<9R2NmdEXi2bbF;Pr#$eUS0UcrT@XAc3}q#y@m|K(fAA*&K`blQf7'
    '&ZN4EqAQYDmRi>$ElgH?NG?<hve1g{22;{1oTb!iSk0+Y?WpkIqe>Q22vD%Jw<VbuCK4nMI1A+~u%0Z<%Sz-)fCdg>m5_{0y@Zpn'
    'B&bkL;5$>`rsA`)in&zDlUmv|oF2sSt3pTa2J<o*B0ftp&$4fYmoTk{V@10Z&Y?)?cow7z%0P8f>YD8-FN-;uG_I`@FN{cr4cv+>'
    'eUG@`zS8%MFJq7Hus}P5{le8zUhD9_w6j{etd+XtWJO$NpvZL+MgoLay5ueVCDo-aRai6KZUbc*n;Gz^LiSXwvQ9$+6$pCMw8lfo'
    'Ft5fG36e>Y{sBsm$j)bqJfv1SRipMMSM6lamV5VptV16W8<mn}{5pD6BP_31HUM;TXr%ydvMRZL<1Stbn)Ttiv?=vGzykxV095@>'
    'fCudX{rKdSd9u|<X4A*dCoA-lDUc&W>V|1vW#OmI^Ks@@e;Cf(`dKm?ga0Wc=wFwB+0{}2CNDr2G04<>mIY8sJ>fe=bL#|PWbsG_'
    'Oor)Ap+H8Hh;|A)QVJ4A$W1~uTfJcgs1htOra}?%`WV)yR)9J%1XlIRVqjS*b=^e9Ph62`m2gWVG!w}4Ga?Ej&Viq*@Lh9hE*S;A'
    'L@Yox0~@-fByDJ^oahsjB$jDqHZQ`xGD$bWe9N3pZ^1?wo5BqcM`rGL6}!NHh`5aj{U4(JCFO#vHR?kkrxNqBHTR-@Zfm21KZ$XS'
    'f~m5uN0Ar<Q7an5i_7&47tw`$9C(tor-@w5X-A9$XSeZILgf>64_E;_r1s53>&rZ0fr{1(sz!FoG-Wu#Ml7x`WjA^%k48eYJe2^N'
    'R-<fSGR;W?`C$P@zj(uiLLpS>JyX_2TWVBrJs?!jii2(%Xwu4A_rXizl+cwIlPPI#t$WLWO(;X(6~LZR0Zf#g_*N9aKp0R?7b3{V'
    ';^q8w<c`W9J4bRDv`a+jx~OCU*i67Ni8d0U!*wY7H~jkXVx~kG<Q626dYRIcD)>jPZh)PIgB*~55^0t811>br)r6zTgK%q~ltek-'
    'o9L9KfLpO9TQz<O=V3pW`AL@22!L;9D6CsD_o6v*WSp4N3C|kpksJ!POUM8?OpB_<jQspg6(sez1k-<3AqluhMI6tRvM;dNDUNIp'
    '-H38S<1SJqDeMzm)Y>iL!jU!@O;HrtDn9Buezg1XN1#qBCC~vu0WqtTSz}`URvVpDTxJBsbSlsUwpc|~sv>gFph!`mtB{46U^XU;'
    'gQ_{Qa%!O^2E|4hMSmjd5-CIs^R5poFVQ@G3@P>7vVcWaSvr_?)D3J{HjMQt(&#K&MZxg7R(-<Ry>>w<0vNP>@Y*vT<}LL|U-2}*'
    'eN_F5xJ5K4D_S(03xleB{Ce}AuDFXt6`rs(w0gX2NgySeWagN1|IZHn1R%e=eRsM%i^X?_k+i>NpG5Mbq~VO7p+MfWEJ@_~L^bE4'
    '{|d*|hFh7w5Iv=MNPmuxB{@?(c|aDd=Bv;?nV;;b6*`icAPW*@c$=?tWromO*YirnQs&Wx?$V=D11QoWgSb(4&KpON{GbAcfi*~T'
    'F^TM0A~{C^Nv0-A>d?*HL$sNEb^+`K21;qVk?Gg9@<_a#?Q7LzpZ!LyDjYlL*x%4H@h)>PT^6XC8c?f;QYACGisKr0iKSdSy;x#q'
    'ZXZ#gldO!nxDK{oR~bZRR!xg60=jy@kj($Hb_|2}lhUTpm=JF5q9BEB06QAwip#?GjPofj>QFJGmoOMPTh$^FQR7q@T`4A@YK6v)'
    'K8VweWv?`9jnfckK2-jPLJzhH50wOB8A<;!R&k*hc354yGG_&~t~|IvJa>tSTBsw!0+1e16Cw~AT2c&t_ZUsUv2J1PrdS0q8X}dr'
    'qTi|xHMz9)>e^Dmgu}GgWxYuhN0j%*Ca0L`(3NEp;-+G`IcupX%K6g_qZ%MpOZW*KD>eNQg+cp;0<B_Kf}_NDIssrBD98bcT})dH'
    'YF^u`xzARxnV>;Kyn<mI1eaG*r7Sv2G7Jg=ialVt7Z)u|WO`u6N*sYHu2$Tz$t2C(G9fB2qkhC$#d}XXmn&ss!S-iC0%~Ni-nd?4'
    '4R&!>yk3hDC>i_Hlz|3-vFS7xY;FlDpmQ)UhynBUbUQv8a7n%DTsszOEv%?+i!BU`{*^?ChJb&YKA;b^6ysPk&~K{k10}yfpM*#3'
    'q!xH%mR_MW)s-ptC&ae)JD{Sw%LH(V!0RcxvNl|eEmKH`BSR?Gc?L#)P#T#U=-_9CKPe_pebyRBBU8Bt9cejhivM5D^41kJLm|A<'
    'q!Tk*R}F*|luF`D8#qCRnHvk?btr)26&Iexq&;IN!J1GCIV;W1xV~y_Uy22qBPJ*QkP+92ZAu1Y`n$IY4HOdr6xmK1=WXTkFam{K'
    'P<*Enc_>`u&0|s(!>vXh>PBQXiAaViNOy%<6{~d<j!2uq!i9<GJ3AHG%kV7~yQf({hHH^WNo6el9t$eOarRB3+5?$rYGAm#BwMR*'
    'wM7{{VFzT(8SN^SeY~%vjFt+iN0l1H2r`p$GU*2wO%f+ukSoJ=2ocy+uRuEAWs}T!w42F}C4UMFKWI55+ssU{^E!%vNtCnr=;R7k'
    '&w*Q5kD)OcTd%MrmCty*vu`|>$A+mNgRQ{_BPv^2+2@rS)vkRhNYLk!sWmQ2<kI4`!B$K;T_#kJO*K*YGD%jfcA_55;<2E;n%*ur'
    'KiL|mi8S>97>nj4n<k{@(cm`9wK23}K(+*@<2&(Mz_>BG=mGzoPQju5E^pz;42EoG4$mOdHCGb#(ftTg3Uv^*j;R69dD($3o$f6w'
    '!~$luvrDMT`Cz^@^ERczU&6QeInbxE>7vs>7$Bx{+yGDY>Y<cY-?a-YiJK^Dgq3?{Jv)pP>=<1nKvZ#f1hh_vM^e8KGx`W~IjQ(2'
    '58?2lpl6_|w#c^>>djVnz7lGpcs1mLS>9eq8Y}DKm`S6M8#5NiKdA6u#O8%<{)fBU*Sl#=oa3X>-Z?b`d|l{S;@=t#DAL0t8|&OW'
    'dj%UilD0_oar~atJQ^vDBVO}L@1;Rbj(;%9zh1*P*$&%RW_s-@5@*k8FeJIA>k}P)klgi${|A8pl7R'
)
_V16_P0_SCHEDULE = json.loads(
    zlib.decompress(base64.b85decode(_V16_P0_SCHEDULE_B85)).decode()
)

# P1: episode 89728046, modal 3/9, SHA-256 262e7c2a99ebb2eaddb8b82d22a3be7ef0c887f948b6acf4c856ad0ddde93723.
_V16_P1_SCHEDULE_B85 = (
    'c-rk<O>Z1oa{Mnm^T6&VDbhEt)H@?AR}v^`8|wiv7{F^7FxH2$Z-)Q7rD1n<Rb^ykWWHBTO3;mHl3nj3->b~X$jD#*_w3()`~9E)'
    '`2Fl(emVPa_wnP|<N4Wt{Pth}_TOJV`10dFfBXGE|M5Rxe*Wd`<M#gk?EGx^^Y-r3+4<`6eE!kq>qkGoeb}B&e)!YpzyEso?(JXi'
    'zW(X!_q+Z7e*O64cKhbbhkxDPzkhdjem(nm|FzeTkN<Z*9_F7uzrBC+>zBcP`uxWuhVlM#w%dOE^v6Try?^-h^V#Ec24BDX!^8VG'
    'pI?6(e|>JS`+vKdkLttQ*MIr^Vf0_4VeLnAKK|)xsKNl(-nixixW9Y7-T!v-^!a*q8pX$-x7!bB9EXKC{^w{gAK!o8KiUp!_K7g6'
    'k;%<Q@#n9*HF@%MgVxh^*xv082Q5fw`Gbv^0s5;mEZ=yHukXwb4Ie%IAX)V6nTIcf7Wu_&-hCPl;q=*07lQD}mGH=^eSTT#yHDE('
    ';eqN>uko$oo|C5yz{@|}BS&*>8QLng&e4O&hQMo0U!d72_R~dYWW1(mqxQpP16k_>*)KP=&tWTuAKHw>TEn~J?<Zr}ZxY%OcaQjj'
    '%%{CySZD^vRUiGN-qxFVh3&e|ei%PSU$prtSJrO!b*1^)r5v^ouH@5Cx$QD*XzvU99)VN2OjzXc2$(e-9!wW=?y$wHcr=P(?dAh?'
    'R#TfTcu|1Mmmh&(-?+8TreZ&_H?wZ1m&ytQ^E5MOhmrVO(h#lJ9=@%(pAFsv`~lo$JbFOy@9(#-KmGdW?Zc<H_iz7oViGB*f0G*l'
    'cpK4oa&daWJX;?$@+s*9ps$&m68`?#<tR;w`mS#t-hU8GU~iA3(b!7@*!c0ndy0$gW!T9xo_RL>Ra|#5!Gxr%eD=kMyNORA@3WX0'
    'pa<D0jLv2_bm9d`Bd;uRW_<etg2MtkK<I>vVdxC%fkQt&96Txf!lOsh=lkdYh1aAoz~dvHe)qiN%kF*U3x_o*OxUhp;@f1W1zz5Z'
    'xh(JB-|t-_*;x`%hXq|6HSWGGI$(<Xlfx1hb&#Y8wy&qu!wuASJ{PCYp16t-04-9^cr-DX)%vFOfs*FYr!bNpjv0vnn|-sR_idM{'
    '!IPj~+VD){1t~8H>#iMF1<YoS=BMlleX~6l;TKyEZuq)muws|Tey{)Ht2%hV-BD2P4sKMh??q>#@WsPM_>L(@na5iqEc`Lr%02HT'
    'eeujvrLWDpcw-+6u5@>=uL_8r9IslP-g5(+^23?y1%NH+%z@WTH)qab8l8MN7k%kB=99T_G<)&Qd2l|C6=c!ISNF|h!m|Um(8;rN'
    'aoi#}CwV@|wOiDM>w@G+6Zzr7{cz#fvYtaP;`?{u03NshXpatFk~pq$<SCCYwW!NGdi`>$6`wDWlq@-abQ_k#^nSj3_*?XICzK3b'
    'T`@d71<p5^Zo7%^32$-(4@jevgr17e89^ZD@bu$H!{FSt`1tAJZuh6{!^7V#tx;zup#b18)X@_hk{$R%cf76#<1=s28L(b$&PW-3'
    'Y5YwiL)d=wXg{H4aJSj{zWJOIlOqg1@>u%`Sc%-4T65>DMamitoooswgNUdA(Na{|x=nqa&^8H{-0d?7;>}yy4e(w?=X?Fw6rH#q'
    'b&FGSQ>*aH=ohUVG<5xpuC!Q>a(Yr<Jk(mz%5qNiW+A8AzR*wTQrpU&669T5nc*xu!(;_Wt!s3=59h^T2*v9c?%3xZtBV4TTbf?H'
    'qBqJphl@j}DwmBfO^8Phew9+E=^X736nk}=&|qRd+?I+tfvXFA=<GCX)lTU0j;kCdMO@ANMZ1VY<TmV6BOy<M_Nn910P?ex1$%gA'
    'g1~tvSt&5TQ_vvj0G8%Op8os0c*FLS*$)fQ0Ir|e@qN{b#KKxi32jXHx2FRpGa*COu?(DA@WO2MoB%wRaCGCRv&z=7+kpzd<tm{1'
    'I{-|7XNHtvu~UZ!09K)s-nZhJ+zdQ}uj6#=E$<`HeRv5B14fX17)2I*Pi?ZKPFBY~0aMeXN|aWU)oN0j%?uzTr3U_6E_1LzXySrq'
    'b|nHF2RxbZ5I_*1d#x#90Ll}Y4J@hWk&>Jd%6@9AvC7$ig&ce;K^6KJYPvej6<nk2;hxW$iZ`WadNhhAst<UwrMM)YtWh#nk3Bgb'
    'n-*fjVs86>O!s&1-t_R{8MvpoD;KFna_Y>+DdNklL{$lq7eR1+tFD?UF0M)vi(q<24KceU3C}kN6+W36u2N;+GI2#T|Kjf@{hf@G'
    '00;vh;nn@Kig&}J5kjtE##4(=l%)MRB^=E7B=t9;CIw}oOW~txeW6|D;oZz0A2Vnazn@^LLaHLUD)U#>ZIjd;%jGak9RxA?MfJ`c'
    '8WcM6TY(YHm$DqPvua%6`VPiug^E(9r#Ociz#!2nT2!P~nR<daNKhpp!sTJb2gzY<6*;eRa6!pJ4EFE2Y1+7u>6jk^O_cH#MruBC'
    '0~AHXUeZ8?jO~rRGOh|zsLMy1^^mhcOIVKI)*XUU$SQ48>By|37+DUls!AqG4mJ<97H{wWvNzGi!%s-=#hIS0mYhdmV{kw*m^INM'
    '_b3Fim8AIAr``l*u<)P&bU%E=NNh7`ff(It-RBlWn<H|)3J?VP!!}DyP8JNwu=*H{j!^8?^twbAB|iHkr|V)DGwKpEIuHy6uV`hH'
    'bO(0Ejq{2cj^@e2BxtfnonBrPqx=3|kqd~YtX;+ARe*`42h9KnV9;PtR4K?hf4QQgRRV?x-hGU|X->l2K-N+S6s)zxDYdp8zP^tn'
    '2760+2QwI|U1snC(4Z!a+#NrfQ$IS4HT3GD0rShlgr(8QsYTlpNF2=bh)IC~j+u44=}=|7B5q5UkS1&Ab&^$<4}3^upc)=np`cO('
    '2Hv7u5-*!Q^^m`s3DUtba)<q7b}8j{6Zjg@+3n1FMGv9q29sA45G^5r0a1ufY}eJRPSAn1skU5SM-;$uhhfW+YS2GUTr7-gjL1fm'
    'B>+D(P!{+$1J1Xl+gIIS9L{X^$h&Fvhh=j2!M5@7M3X$?A$Lyvw$!)*lb1!vI@zPK6#xkS#`cJv>^E<JauBI1;X=bU@@UN}!l1yu'
    'IIo;=4lbBht4TlZ3}S&(O2BTuQNu_@2fe~60p=VCj@T4_=!fQIyEdr|GY})8@jIf-AI`#b(f`<r&MVEGdh3HtN2EiLn%GDs3#QRH'
    'Xek_pGFXl*(2~z=1T9mE_JI=d<BS9Bf=PW)8mjUatep!w?V6gj+8$B5Y7|+=aJ2j#P*lREbTB*W;rbvzXHd4=&TJt>E_BlrwDB1a'
    'K#Sl&A6IaXQ??GhM350D&4QsF$m%G_!6IoP<p8=(Brs&Uc$|UADiPWQhm_ie05D3nu;a=&UW4BNdm{?C+A1~)Fo*NJYT|3C=!Qpl'
    '!*|*Gn?yZm+WAGm9^(&LtX#r3F8UKzLaI%=Qd#G=G~xvr!vp?KG%?6GBK=VNOWZ*8WB4IrrtCONXK`3$EASbl3J#s|k>7%qVo`SO'
    'qv{WROF&vEo(1tU#NdNggLF^@AR}ZI`3wN!0ag<{M4R|Tf};3%S@R!-thI7;FiM~viZ_>g&6NZiixg-jKGrEk|I*YxiF&W0t;@Ju'
    'ML(c`+VNNi{RAi-#SAHnWjT8QkPbEW^<FCZIwb*YPMq637v@RH!tybm)68OwG!44Yx;Pl<{O&7~$+JzgH`D|&wKC2L0ndbTldOrU'
    '7$>1@ZPPXhVFZ3C=@@NU!g!u@sK{;!tn}9EKhuH=H0b9;1u^L@!C5ztNs>hLZf3e?OeT{Rp(*7p?)}9CS3u?j72-?Q<eOn6Prq&^'
    'd>k6Y-YitWk_1E!&m?s_n{NaR`Bs1r@I`+Z@VWu|xx73Hn6(50gu8LlPi$qb9+#nqAjP9uL3UcytyABqA|WG0Qttek1VNLd9V=lF'
    '35?WLP3XnwBNxR27#T|8+KrhSffz2YR`c+jXizmk3V-H%uWZOBm8%9m$MBJrAYu6x3)fMSPby+&02UYaqaqTF11uLhQ2sM3%m4{i'
    'lFMc)jt(nAmdq+iAOz486#=)`Rz}~2MV0nWXXF9Lb_AFlpXnH9v9=wN%~?xK#=8I|Y(lI0W(?3`E+Y4ib4{olbkvzt`8(`fQfc{+'
    '<CrEyz_PXW0zeQ6a;(l3z9^t9FbA#Nm+b1H{26Wc7^K-$y2R>S62=qmYP!@%-=$JR1PyegqZUamP24;-a^6WoX4ASVW<_!nYdpoh'
    'UI1CSAbHK&+nT0!V#Bab=ALhSxirqB=Y3i1Em6;U%_KRtA`bM-bQgWd*1iO9kb(qRRA+qjuA3D?^kz|-h-Q;|&}~r~gg`dKCkkgt'
    '?v%Sk0|fdQN%SMSU5ElgoaeIB0FHYE*4S-nq4R_)G@FFll~1%0PD!XR69>S>+=Ov&_VdVy5`Y!A2Z;pCkk_JmmUaSYxEer0OC}f-'
    'z^!nh8HcM1;UD;hcCjV#ki<~emqMnI6;vQEc&lL1?h&AhWwXH^4J?SD5Sa=R`xE&HYi7<pI=gz#tMayCP!X0-9wA!BF<?<Xes*&e'
    '!59$2g>tCJpcXOa^nEsM`jJ$bmd{*_8Od7jt(dEZCh#XvACU>tLn000=+8d8PHCv+T0b}JRdUj>MPF@3&h`PR<MNJLlRN^yg{FL9'
    'aW%|P6PQNZ^RyaD?Wv?WO3VrYS)>{37B?GbTC}xfs%kg994omM)4Q8K5$??wH{s!ZmYC2C2=tMVDZolUP^A}&D&Z9W`QqEyf;N>z'
    'WSS{NYoa0C$Ycp?oPk@bWs3C<dm!CMu%$%oKsh-QPpDpNp0E_Kcx<h6*HEU5=Olz$U+64s0_=d4Kg*Oe{sdSt9t)0D;<bIW@C|`E'
    '02BiJC3{AuFYv-EEe+bBG0dlxK%gnXi+@jLcCyeX0cj|+i)Em<5!b(>3d5TlZH2C)YzGu#-P0Mr)PYS>GAIfqXgDDfc>-}pd((L|'
    'YFD?0*bJLsgVBRLIj=1P<Dh^bE4%A@SZS#r_+{d?F2M(VI-`Xk-9d`Xz7;sxWT}nVNsujIk!A%@goev%m{unu@e?5ua~NGjmrjbR'
    '@}>nnW^N2#6baRIy9NwvFxd9Z^5()Y9ebeW_*XO&SxmQ8=k2Br{E*`n{6sWz$bUATGy#?4Oi%*3(bBkVd2g(<d)XP@x@;mwP52Mg'
    '<43T_MXJx$MSq+~mzinAxHikeJPGiif_PDtkYNY6Km9cAc4*Y7rdz!^#a0(|lb}s!TTOM<R>mk^p+<m<3h703F*jhf>D8)9RaOEh'
    '69Z{%xf8|9=BAsPg5Mp)E!pyZmX)UaR@|*i#4WJwfqdezu&c=y+j}CDk{)YbQZ1FeN*2>^M!t;B*J6PKePJeBS|;Q$A|Q}n2Mjd>'
    'nQwbBaNX@mYWS6oJhAiu_o2a`(HbA6Ua=y^M1y_)lf{&D!7E~HGx#sv;SBP6^`sKV6NE}Svi=cUO;g{Zpj8YnrwGaHRunPFtxN3H'
    '({T32wAF$XJ{kTTwNi-b#5|O>hHL8!=QnA5G8wT1uodHW%Bo={N+n$ZD$5~7^gU7xW~;3DMOoH~hG&wkJ(!4A66POtGCY8_Gnr*+'
    'C0R8u0kDgs-1!<@0Bk{(XE6v$Ojc1fFd*G=Wh8+lZU4nv#L^?n4UA(d;>C}Wx)d-(2&=KGUw`Zi=zaC$NEP~0chpmYgA3+cAVobV'
    '`cr5~&FVmMM5hIT9ojt!MBXGjtN=Ti?DS5p$E<|fZQofby<>tmLZyvY1*tXI40nP>#}rOu6f>&mQ-Nn1I6cwM1C1jUI}^E26{Au5'
    'RqLwvV$KaLK<rjp^9Gj&4G=h}6zyN8SxQnECcdnrm|u+BM!v_yFTT}LE=5_KRD){L>=nF%3JbB0?__Lp?j;c|m%6h7jV^Fi!w+{`'
    '|J3>X5e5v|cs{%iEGy+0RWX7^P}sA_Y;?(&1a=#6*D{iUh}r>!wW<x~!G>^=CJGhOi2)Ezt>AtI>iO(P>)h5bu#rnQfj^rSEX|;u'
    '+=<|Te}JV+(t<NgCFcak1Y)_Dr9c46jm@HW;H(H-ScxvjcG;NgMSB<XW4)pM<awZ%^*0Q?>?=TSP6@N{220PNZ?rJX!sp9{j`Rrn'
    'o4{FU;MOE*^bAOgPL;DcI$q8+qXgCp211L$zBwgNh*LfR_`({X6Ro3NL7J+F$*`g^xhTu&87FpLXfF5(dx4jf_E}cFD-r|nvBZF-'
    'j#I^0o46oii*PER6)VBSthb4WaO)<}<e;3=WmJEjb&wJOkxr#$l6(v8JQs{x=W<Z@pBCFW!r_eSW?o|p)k^X%=Un0|bvv<%Bq|Jp'
    'pxAHVE%t9~!%Gh!kEZHW;9a!TuCc(M%Gyu{m4>40qSJ{xjdt`D^`68#Q0ai>7RdxHu*SzITiHwVtBID4tin{(7Z>5i0L;oES0mwP'
    '1Fup)TP<kRichdIQzcvy{ep4Sa&w{@D*Udw>W9eGY6`klNi6`sP!nv!Ty!UE{S4(yUI7XJhOt#|Fqol4J141hk_4kelSRaTT^Lw<'
    '@3#~<*&mcvVvBAP@b)}wt885qh1d6i0T^oYwOqB8T4Pr!c&^nmAxDjf0VPtk6<DKI-^7?~w8Bov|CMx#jJy)}1!)4Fx^vT$0C-oG'
    ';X@ibYM``@V`OirlGsDVV4~76r)|9oky6K1QTz$JEo_=VCDwY3N|}(RdIVAl7|jVi6DIF&tY=^kpf>^Dh<TX{Bg|jn1S{(+CEUg~'
    'RsvxhC_vPxPPopGlS9NQ>Sb~PbR{J+1Z<N6RY!y1tX=64a1FcCV#kf3A5+vcpKobXLV=OQsjggt$XW_}Q-eY@t)W_rAld4%M#edS'
    '8?3fW)Voo1?7XZGWjC23ts|5;4BJ!47timoXXQ4ZV7b{FND&7L;YT!kT87yQ7meTVH_kn~IR%F&B+#waOls(Rk(mC$>v>jQPZ&Nc'
    't;}9x{S+X*5ai`$5_=&0oFEP;tz_FsJQ|@&<6i8_t;)*vQjl^F6Vm`Ll1tZe<Qyq;?9g#?cRA83uW1miN>rXy0SEIx7m|?$tR1LC'
    '#<Z*)pwta?X#WhQZ-u5N!^Lzc0wLj$o#|fztW7mSA^=Z`6eog}7kaozC_8Xi+|@_F^~_2WMhYb@V>mU~2mI@xsD%b=p}yU1xxt`D'
    'qta@$d$*++`w0<xt2Kf0ZUU|q77sb0rY$xtM{tB6UlL$#?XU;e_2zv+oK7y#N$OYVB&9{jE9GA0nmgN+j;<8*H^GDiVB`E0vZ}3|'
    'HoAhqUfnieA-qrE9NHP0H~f?YkeKjwAYn$NaK;DhdI9>+bXB?_IUr)DV9u-OL~cl5WtqeA+>D?Ka#Lt}^>QSRy9?js3nKcWIJI<A'
    'bZ6$3Z#ErvjEUP2{X@HbNP9My$DweIuFM<I@DuVNr&YgUD!_IpSmB$iDqHl=1<h}yaIHf#_?u)qQdggdg=!7tX^vbH=IBGPLM$1P'
    'g^hz&X9I$L@58iWeU+%dl9XfDT0@3S`d0PeFu-P`s-bNRBr0k;>n4e4UN$6F*u<p|$r%af1Q&w^8=;mb(lXziXmQv}o?x4b)ypS='
    'c8VgHABVUC%#7yZ?P(xO8{uIhR2_je!2#BS#bS-8nAMW1-SlqSsYXJIf^C3~ZWVk+EP+<Ve^Rs&M4_Wo5*Jz;_0#(b$aIgvf~P;)'
    'WH2)()XWf*IGj<${#h*=T|K(o8B$j<PGp$nm4m+`rK#H!Rdj(1h$KojKPtTzuk_M1?!LoHFR`LkcdlCIXGUjX4&iV0F08oum#P{u'
    'D?V*)XD=3nC1i%TQXi&etN0jl!muwK8}}A>j#gFa%~k4SRqoT}lKHjbl1-*Ju*JawhmVVLxb1>-u$UO_vVL9kJVcVvay!Gv#9ZtT'
    'gMVAd#V8ZhLZTT<Xw%YSU5@_~Ws@R`BcH2D_`P;H+`8&CyO+Tq;cL&sYvvbO%D3lmm)vnMx(b>6N=}M~>~-A3N=7c9&RuOKWkRT<'
    'OXTdg=$lS2rdJ6LoL@gDx67i^MKb>_0|0nP9ot-1Rm`BVDg&1TaEt`X5}LD4X2MObl>QQWkY)nGY;`GOhLAXrGI0K9MN2}$i;OU&'
    'b{!nDn(B7sho+f5Zl^I;5PQ=M8>%wd5iFF@hHmO+1rWT|^$9DqChi8opcj{4W(BU_JGn>T;l5N5h*Nt~s1H~q^KUN+5K1AKVv0^G'
    'MPCr@5s6LA%Vd49a&(|BWTxqB0608UC$a=nCxt{Md*D!m=h>qff!Mpb-h)U(G4La-z?0r(r8<Lvy2Sds?Gl1hf0Xa{ky)lih<3Rv'
    '!s23Gbf@`5&<e$5IgSj^0H+c35Gt0%--3WZ>%A5Cb>2uqJ}}Y;r<IZ5+)&ufZ|9gjEFcR*vv`=PYTmd*Z7$lSnf72Q1GdaTJ){uu'
    'hT^m|kivE?f%lA_r!6d%hxRCFTR)ChAknz9+NINXS3#k~zF$EMrPJrRG;(86QEp-R|7Gi?b>Se!V~J~3J)|sF{^J)o8E~Y~Mb~lX'
    '*Xprit%USg$h)H+x$VO)jjw#*nv{1|&)6pG>W6ktx%K#w)z)7-2f7L8FT5qWkAeuo%?gNfdfA;!GEo-|VyW=w65(N>5~<?S`)w%#'
    'JS7MR0yfULm?(8dT_(rh*ZQ?5bezRf^=%!_*_1l&u!i@n<2mgj-m(1`@FN`iLGm@8NeLQ7g`P*75<$gIKr8%<_$aE~q-Zt4&FO{T'
    ';Tq?W&dwU(H2#&<ZfK;>$zDT!nl9|b%MnE`MZEGlwE{O*uI^2?2|Xhf(}2>ICaJ20u~QCh$%}|1Rq8H6=XU(Kb*jDRsc;aZNn}F0'
    '1pN`BMG2f7XrQ$3jB@N(D^t{y>#olAOtg;cV8f(JC9dPLRd8KV@Guq*8_)x(Eqrq1p)bOU4H?`tjCx)IMbJjjii{1I*N(F;aIM2N'
    '{j8?bSZ6GTLefFuso&wk&c`w|HP%}bQ^<0?ZonJ$@<at>i|hX&b)s5zm`uG^8!jx!9s<5eK@Q0N%ePQNRwd%-v<(fNNlg`HSEQ&c'
    'W!EAtOqW>C?WYA<XvcPgDQO+fQff7<=Txb8RQT^vCyOZrC>*%UlFSPe36d9_rScV6PnPCoE%GEl1E;V`NXDk#!bva*DpV81&J?(*'
    '#B8i%E>-fRo;FRV2YLLe(2;w<yh?_M&yuXO?AzfbEUV#I(Kdy1C=xoJ2dRQGP~DV<W;@EuVnHU&Ym3ASFmI2IZ!@}xraw#9pZBoz'
    '9plT`t2=Da&S1aLI?A>V-<Nh*OD}7sFFCP@s|*ymPQpll@JdbI!e3Hf>YTx^Szo#Gt}`|>;8BI_saR#5rUb$&X%Cg|(6K9_NUFo^'
    ')bxc4%#Ex3eO}K|Nxjmk8nw5%>Lh!P+<WiGI`k2-QE6EQ_G$DcY3$Fy7JzOJ?G(UWR;AVtyo1M6yh@+cO=;!<9vEl^pze19Jjhiu'
    'K#FW*P`_8WX=r<#PZAJ{gq2YyM}E{T*~Dk#%&oo~&fNN?RW=6yQz+2C@r`};3;-r?Ko>E{)OwZ$P)fbvJ4JKr1Yl(GNCr%X<xQbL'
    'Mw5tk3OiC75=O{PLN#0U`~|2Im>5%`h<JSr>r*>G?HK~AdS@|UR!Uztk?|9EBw8iH(g@82^8Ad5!iaO=t15igT$xKoK`)UDP~E_W'
    'X(>e;S}7;`2}%;nv@%;4aWp-B8ezR<PN%ovBOrI#LLYgsN2(Cwq-(hxr>Or!w7;ZUaJ5H$2;@|9Ubg36bk1#Wbnq)Nk5O<{*7YnB'
    'V<2ixV|a79p6McL$cKd|RePGq#hiA;j&=a!A82v-{8y>X;F=Xh>&rZ0fr{1(sz!dwv}8EKMl7x`RX2JnkKoGIR7KYUG_7XYz-F48'
    '2J*uKjDGQf3yngk(0it>i)Lz6a6KSY(2j#{8EDeWNn8$XC81u%k)&JKr$EGX%Fqu5u#+f&iLw*liUJr&1Ipz>1o>FJou7`}Q5j_C'
    'N)G9#3H|IYDp>$F6L3tTjYQ~h9h&|P-+p*8Qz8rs3zA5^Ole9r{3EY!fSrYl9FTt!WtH{^+-RPw2}hd;;nqGSiE_O+(J4y-w_;DW'
    'YW@=L!)`9~lWe6C0N>0}ShrN}MSJ4NI5DLYUNzLCI20V0kO8pY7FCTI#rd5&Na}eBrvI!$63|FR9M80}FR<Auj%+X8h;l>oE>b5c'
    '91~pB#x3H)kv14DQB*P3?N2{V*#VZmqW5A&ZF8)kfLK+^yfLwU>$-`~DViApF`W)HfiG53m8yu`GbmCN=qglU$~4Awb5ONLRxT~H'
    '#Gu$Iqv%gWT_TN$VcqrK@)E<-*N{@bEelv=m8Fweo{=2%1O>BUtWS|fXVEGOhR?O?6T$9{3rcyYK;sYIc*YBVNh8u%yv%PGb-yBE'
    '5v|FZUWpfuI^!_iukV46U*EjPH`+y_3s2Y@TD{)2B9M|yG7C(3{AZVb0+8R|y*s`<i_Le2m9*bxKZ)c=$-o&sLxH?$Taqa9iF(dO'
    '{}ryQO}8?AA$m*kl>QtaOLC@o@qjE?&8^UWGQZhV7CMrdAO?vlyv?6;Wri?X*Xv5fR_4(Q-K9sR7Eq)`26?0Go;R)_`9cK@12#x&'
    'F^TM0qButZNv1AI>e9`;hv+c*?gH2gER-^IBh#;I<&k_jJJ)K&KKqVZS2%XjvELyx@m}U&yDU&Ob)Z&{rA}tlisK%42~)1!UaT-P'
    'caA8~Ni1V7qJ#a{RR)onb<-k7K&=NX$^1WS*D!cKX>AIf3E{0>6r`{NU{{0OaalN?aXrOF9V%w@5(Xn@yILeNYMd&gE5!s<t<bsA'
    '7jb&9>?@6W<21#YPnCaH=)q+oLM4e<M$&)ERa`iD(~VaIcNJj`BE=^UE|ADwVxkuMh_D2t2h@ZJgoc(Blixj76L73s7`rJ}0nCO-'
    'Ew1QaRhOE)wDs=VQo)47wAW?5Nft+x_r@-#Sn1G}WfJ11V!Ju(sVK_%(+Z;+AXO{)2?8s%{1K%=`=tV{VpoEr#7{Z_U>YdM0f}8K'
    'TMT+$+gEd+qhKpRgMoMr!#D^oTT-PhIx8{^8Ul(lVEV%sI!&6~MBH$_dn$0n)s7psOr)7xCS>Jh)Q<$K`0Q!-a;0r7IQ}e2K%ERW'
    '8#ijK!7h%)>%AC>lCgik<$O$MjNnXz@(fyd=zOzNou_;4(eg@KRp;6zP#a-IV_O_y*y67wIW%l-n;M`GtQ50WGl<XF{*DwpWj<o}'
    '@>T3jSu%w(R9BST-w-?6AMlAD920njk$0@h`HG2{O|0kAT^T~C&hstui_*x;Ko>u2^GOkS8ll!08JRvk=t|4^PW=CBezvZc8Oq+3'
    'Hk~-lx<()*oK(tO+Mo!s%-mTBUxxw|UJ2M)2HJCW5*!JoZL>1$OrWcF@}<b1xngos4;d+q*r#N$rGI*xus@OZuZVQgtZplFhtVbE'
    '_TgKVwnM=oZy%Ey7;f+Je8+ooNJQ#OL8yy$t$Olv*G;G)9R^DdCUWiUR%8#uZ>cyu&BifYc|0m1W7+mtJ|WJkZxX#7$VO8K!#yQM'
    'h2WXsc{l;tb4I61<r?oQ1)`<4=}}(>F<;Dtn@q96MU$Wjm*2`T8$!G_H7iiOchw{+9-U@#V#%+<HV?l3(6p>1u~Y26j`Cd+!7RQy'
    'xn9+C;1)JwXe_<f`zcAyGM?Vqx87B`5yX*+q)W0%W#yb#I#Rp-rO>D@S30dRPoj_(+Xl0ka&b(k7n>TQ@W-Syu{w!*?uuuAc4{%Z'
    '5d362nkI_U17IxLlWb9tS~Y{mD0jlpTK^mxYo1536Yu<s2ct_I@bBrA9NKO39*)dc$kyTTdNEyLCD9Gt&mg5x2T|FWI`Eu@9jMUh'
    '(Xv7#U|u_W2~`;$ES6@`rqt$3_$z)7^cifr#Waush>07wKv?4Xu~bCgbqcIxJJ;=ndQ$t9f)k@g0wffNkASl2@R2qz%Yz<ZE+=L('
    'RR{YA1v3LfwJp7+RByJI^R>`1?hq(m@HQ^W*CJ(t;+jdTkXtjB$KPx2U&QW(9sY;=yVu)kTbsj2Mh5mc81PTR{1X4Bx>3_H>a#uY'
    'utuXA-3w;KBkhXRM8}Uw&2$gc*e$)%CuxK@X2W{L#=ObS)jq9t!t`1O)6R`i-^}O-DLQ}rfA=I{X#'
)
_V16_P1_SCHEDULE = json.loads(
    zlib.decompress(base64.b85decode(_V16_P1_SCHEDULE_B85)).decode()
)

# v17 core: latest-train route selected without holdout access.  The complete
# trajectory is never switched mid-game.  Source episode 89756344 (Anton Tikhonov),
# canonical action SHA-256 c8d7441f644677064044280b8debf230755c57c43e374fd0ce13572b277093c7.
_V17_SCHEDULE_B85 = (
    'c-rk<%WfRma{L#rxnRHe@Qx$(oe?fa6e!7!>j5zsz}GNfTrbAn8UA-`#AbC>Wn^Szp3`J&tX-qn>^kq085tS*%m1AH+i$=B<L|$p'
    '{nIaJA8tQ>JbOGp`}g1e^I!k#%Y!c;|MA=J|MB<#{_^>kvya=myR-AN-A~)wPiN<=$MgB4h4bjAcMsdM$%h|4|Lxb?_wRnb{o|`|'
    '?sxky{_*(ZcKi0r!=JZz_wUcnFK3VUKYRK3_`m1lmjCei-QC+?zqJ3;=dZ6Ay70@{Zu{}m*O$J(fB5v%+2gcRi6-sey?K88;o<)6'
    '=Qp27t3JGY^ViQGM!z>|*1k38<4;dboi>oP%QRqV8gO^}X1o8t$=m1Sm1@PupSIf%xpq7c#PEf2V?N%0-oM&5Yxaapo@vEu-ip8c'
    'F^`isPbX+O9f$4he&L`20m~dVV&M+OVH8IVdUI>0V|euR!Oly(4xfJvF265p<o%~%$4%e;bRdjh8UONh&3zfY+fUnv@ORPXFD&$@'
    'LB&Hl90Z5;hT1Wm*=5V}8oh`NGCY?+E7T=t24_D4Xx)#0A5GN02g0H=Z`}_!Tyw)z3?Jf|$gPI=hrdtSu%9F}BkmlL_5VC(aMBM0'
    'I{MUbmaO6-wgWr+K>9*PZN8_K!JGYDr%eV1e3()=-cO&{6q-S_Cr08bSh;;!Czk19Uzl+n+Cm069<;-<7=~)z95{IT)UAbE2pq|5'
    'tE3j}tS|P7J)PlF)jEfU7%WHAPks_4%qYh);wI8MzQefr3>L%e)hSExVC&r9-EH4|`t>i{hfnYB-u=sDNrbk1(k|p}0=E{j-9uAJ'
    '?RhW+-QGB|FrzmOZEEc#e$ni74K2m#qFbk|-M0_-9}4dpwOHCTheP$rrN#%YV+F@f1UpK>8<~-%SyWON-3vrnInO5}@6zmYXzCMs'
    'qwUma`|F({hFPvJm1!uoc)tg+^J1IKT`18Lo}o>!Glv(kSBAfQFq%65Fl};T{FNpjes1=+6KQdI#Z-QI7=o6uuw$(JEW8yOj5Y9U'
    'f+dwT)v$G>A};HfG~}3730rq_6qC*aLR<09f~^<(Lfvh>82l(1dtk&95XJ0`iwOKEAp+F8ih(?(BkUsgVWZ~Or{E#=9@qLpV6mcA'
    'cyJr+FzLAEB4rj1??gt{(;ri2;bBxjZ|0$h{2jy~Y3p^dp3D#BtH;oQ9A0qnx$HQiH&x9?U@@g2|1U@1!+rTxU5qnkr#9JU!yNQB'
    'fDhi^xBJ5tTe1qp^#R<bc*MEIM9%9N;EO<X$zbW9){GlwZcKQX+rO`2>&+XsdF%@HN!DKo^@;W%xPYh;0=L<9354rz9Wwk#h$(D3'
    'L5W8)7im1`o3j=vBv*+{QiO$>iy5i{L9F986El#RN-u5(hC@`e`MUhBX7g#MR6kQtguT97h(-KwY`ejcK0lH%Nusx>1PAU7k&JD7'
    'OoP&E?JJp<luvea1Ck9%54(|0!}0LLpKc%iIr`iQkwZsU?88jq`xRy^3rt0?a|5@kM9{GYjL*wKfa>t}!=s^b?pS>M^l-cTVf*m#'
    'w-GyOIv@g95qS#FO&z_#iL-+*y6tt3Dxcki_JFl&bBmO5#KylgS_s>Z9<?f58Qg4kesA8VMCS-mlDyVF0V9zcQ@h+bUzAdDL))7I'
    '*T5qRtye}9aDOjlA4_j&>jaB0_vr*N@+|=gSg)e-y=()j-ACQrmt5B>{5<+Y3&)gxvFN&*xU+tJnxvlHvpN-{dUYhD+Fs}{Xj0o0'
    'sKUrBiX$`ZWv7`eQ>%514(H)KYYw4!{oEb<-ti8gyuHxw5d96X3lMY|8CAK+f2l*<bMRBYI(6r;{y?H!rw$EE%tzr;J|{4BfeoGQ'
    'MwlyLP2O?h$Rt@#BVMKkaq!%Rc~Z!Mt<XL-JQ_fL7P4S3&vXzN?<9N$`X_QkmI<&d#~t;+(q4?rm;e2~j8cm#9}Ppkd~L(mEUv4Z'
    'MFJC4CA9HaX+TZDq;9#b%fbG@s09~htK|gXxrCuBB-fG*;<4F*lIH2#``iJb`#aMln(&t~cmQA(8tD!3333eZaJ>%7u`At&oBNOu'
    '7~qTW_t0f57@AsVN$Rl<assZW$BQV9CQE*$<eM29hVQid`#W}}Wp)wpLtKO`)?ta?Ot=WZ;y;qw0_BR#8kX`gnCea@%gF1xZ`9;<'
    'SNO89kb_Sps6zjl9JKDFB=&O8X3d@5dM;HH-v_+e5?YdP_ShEP((%CyRxLD!K1TZ8?fbXKwb&VyCoobL7kH3R2HQC7yUr?iE&Af`'
    'BII3QNN=j{fvYFQ{FR<QLd+#eg7M8RgwKP9qofHo6`>Y}patVn{Ch?1h$<@p{{Yfn-LKV*XU3#qYp3R)C=vQ|$}X6ZN9q@$rmtmj'
    'O5ss8anX+Qa5b~n$8;IR?k51M5SK`f%KWCf>1~@bB!M|G`9bwO3e+gHv$p~(nh#~!WoK!=z|I{Oqg8%NEj`6H)BpyFJ0a?FRYwsy'
    'AqtA!bZC@cx{bxp8a5)0)q&N_K>nVyrnL*L9rJ<4q#A~fB!1)?D9W-;8+zSz{vM(2LdX`hxs!)dbw0a$p}IdBxpd2d>ay;qLRzVs'
    'OWHh+qO&<%UzJdl?0-HdTfDpb>wbwI-6MtUUL5LCdX`YGhS)*Tzcmgcw=x8_m8*-_Kj8$0uyCUQoIgBbEVdciKo;F%*=y_A$q~t3'
    '1yllkv27(LI}4g*Xnl-FM>zH^TJRas;z>}~g)L^>CAR2*gA`oR$|ULc;fx&TWi=d@Cv%<Pl0AqY^Q;)%AMmnVfRAO@Rm@%mI7zzE'
    '41@p+8Z?T^5?SLfmvyvAKoh~;$M~COC(IYfQqndK##$nk+O-{C-v<_hJ*B*j88p=nGq?aW_z9zVhevbnM~AkCR$bI!es~y9HEKCs'
    '(RK$C1M@s$S76X%wz}PTsM1~$%cVm|qqXxy$tukUz9d?p8b(;*pi-O$o}ydgFPk~_(7@Ufq>W{S5Bp?xDCK4o*c#E^?M!<`FQMoL'
    'lV}q_E+MD^QHV}-*Tq#WPAstA)bL*#1Xy&TpW;F;Ri}R(%s4WtF(M#Q;sAVTpe*of2C&aZ;Ex02aB8#1-c8Ft94DL~OdD@c)X4)L'
    'a_jhS$7(mA@-h!u=YQCmT{)lLI)7ty8yneg-~H$yQdJH{jBThja27cQH!${Ks+?OW>=vs#QOHw2Ocr8>Q|7>KK6gV)MVq|>)`Jxu'
    'h>(Da1(9i*SNK|`0?|OAgu3*IM*qA$ri1=<DmsrexAU!8Hf@oPN@@ZmmB5(B_Mqu;6y0E9v#&hxbyK8|%ni+EiKd29_*K5xN~=Rk'
    'Em-R4^6T055giPg@U<GNlTIZ?6*6oszb=X@jFezzTRogK1SSo#e%rY+gc^nprGkuq3^cI9xZrH3L?3#vz*;!D`DdUAT3!?jVGg%O'
    '9)Jf%To8jn&~#8asgY$ev<Vg|xe)>0q-mCk$~s;p<p8=P3cA`THdX!z`$>V5*8@kErwfiOmZy%6j2|+Wy7#1A^aMnNP_WXaXu=y|'
    'h0MYOhE3ElNJ}EsY5PmzK#*kk5D{2*oXm4{^kgg09;Dn4ofwjzf`zi8?AS;7A^Mbn=urkRB$kEfZ_vz<4B4%qPR?*8l0J>>O2Mb<'
    'C*G*5lH$E*_r}zrZAxed?FC|_c;>lrojB?=*a+B<U}s<fRP--%U#jR>QUf>K95Rg8mV#A)1G&S5J~3WQF)YhmU`|;8Ovj{?F_d}9'
    'C=i>y%BmRKC@mjq=4_6O(!{{jT!mT)dJ>hg!xp8rPe{C)`s5yFrAl_D^6~IFXr04J7{_F_2kv=1kBduPfs|-F@J%e#ejR@pci9J`'
    '>^wasua^>3)4C57TR_ZGoh!I{Oma7(>z?VcG5JoGFDK?Q%@&)dzPn1v$AAD2QWcQaT2`VDoo>r!(#rD>U3rB*N(jSia#>cV1lMF+'
    'zj`HT2K><1^}IZ4Z*NY3wZhFo>C^EEEJ&wmmPVe|or~h(`@;ZAnY3p3_*9Y*+`09$DFlI$cKC*&HV`sU<*YjpU%E_<!60<i6sq|k'
    'Bsa9r4iCEqX$>SZX+Fi3ai`I6U|9^0tU3w{;8>iHGLBMSHp9!f7$FtDU>s#R5rguxS>gzY?2>3XQ=WAA7==>YXxjv+6_q2mr%(o|'
    'WZtAbq8SgtStJ4wkN0i`DvDU|tT-p5aDZGoR!^!IXlUUVk@3fAD>V9e5W!TbKx_k2k^7KYnWkL8{JHi_MBvzQ{LXb`Il59i0ND{k'
    'DL2|IF*viS-7K*tu}H$%OXn6b!4g`NR4c)tp^LQDA`PaA&c|BLYgfozT~|u2NRwivuDG=dz$h0NudVhb`pja}hd;v$v8SP+KJ*zq'
    'g3Mg$7lkX5Z7bq6uS}fLn{3yYVD(YoK1-S!Z@tTANfNz0l_unQOubVAs0>dOl9OD5H>2ybNa=@TKY)!VO`Zyl6-R4f(*ET|OI!{Z'
    '>k;5$=dOh$6icnIZxUi#l)w}1nNZ*+c6(_m*tHH|?)#C^L;z!M_ZxAOAtgq&Gi?V@b2a>f=3Fo;fSXOjW@U3($ru2Zq8)5WyF{;^'
    '24aIO{lX2#M(DA--pMIs_UK_X1<sR0y0X~;iO7OoQ_4vSiYlx&aq>XuON2R@2dI_t1(*|$-`$*UFvfdu_8lr_s5y!`9idg5P9;^v'
    '<vkZ&#n?YGN92BGb0O0N>;w`lGCn&cFf8rfSdO24y8rq1)4eS>!2Qi!@#%(1O!j&p0yTBUnK~f7T%Jw4Ob=MES_JW8G>v<cvQ+(0'
    '0~9rBY&3^Y<3<qSO$HHWYY(Acq-N|E*F<NExk@7AMR|qstgqx$OizCL#$#n>J<+Fn3gz0teKAC8{T-0{h5)Jc%i&QaU7G?N)kC_N'
    'IT^Zjt_6tos(&ELM=+>_KS7x_65FSa&?J@x7(6!Cxud1IjOZ!><AAU}i-t2Y1Q;ITdc-qikr{$u0F(pRICk|<FI|2CiGWgo$DlUM'
    'yOn^NDFciDPNjFUgebvZC?$)9nkiOR6_??ql(s?*Q#J#6DWXh`AyF|w1N@LOl<x^VIWVgh6&&t0x>`46&9E*vEP4<Sm(rU8xork^'
    'QU%re%AqNK;E#z5TfznURD}*<xo1T)dn<5O$+0G4Cjqp8|C!ZB5z0G@j0M?F7c)w;3>?M_(Pfov!7-RUxX|N83`-PA&~yd|&}vx2'
    '>imU$S!bXbyT|5$SJV?(xi_I8=NB^?!x(-=qw6}<LpEpdiHPrz{}j1f4U^4@p#)^3iErd)4CfMC66-pBsu_jg<u$m!fB?1&r~cNl'
    '`8&J3uFdiAtW^iBCjlx{&@FS>gwnI8!@#ocx5viH)$HoPRKf^dArpX$6X_XoyD~gc=qP2m5$+baQvs^kmiXqE8F5pqyw9>^_0g4|'
    'mnB{n823PP@sO0&wD4x4Z!4fW3&O+KedHoYf{$IAQVNwla*n1UjZ7Jx$Hlx3dS9~XyVS402r{JS2E*4t1l*n=T<BYbdz3^V0L`LL'
    ';A=F!N6ATK6ojuK?JE<TU$^Rf@J|+?(lxcbs7Jc&8YBhlktPmB2w8RHwu)$Nnur(0yJBQIm&-}cL=nE+x`banHfPUEn>k34m*MNE'
    'zC%ni=6>ufvoPLbj3NRUi*ZF~b<HF)A_onsm`93ed?XjlCRy=Au`D<Zw;LOK&=D;o%-_SLc>sZDy3Eo@vZP%CYZqI&^Ch+bh=LN&'
    'VrZ1!Z;7_rk&$GLwE5Q#i2at@jIQN!!DNw{cWN1Ri_;}AO(*H91b-CqE1<~LkJH4fXB=Bm95ycKZ-L$Pnov)n*fmS?$dQ<ii0sfZ'
    'Ou&66@!<lpL*vssHTAOMFSq?>A1)JRCo$t13+%Er70v|ndMU!j=wnokr-HFGuzI4I2l_-Rb|$i)h>>%nLoP~nY+XEGjHH1fVV&~Q'
    'X6+UL2Mcu=IG_~Gai-czQV%9R&?EQ03tvN#87G;ZnsmE_I#{qbBOb)c%ahj1k(Y$4T&nv91h>H14Il2d{-*OOCAp-~@HDVQm4i#g'
    '2o^zp&K`@=<z5mXZ9r$s$owH{2N2JyR<j2a!iAV91V}puz%hZ+u*{7&Lk77P6ZqOJnP>*q<d!*y>jRiu691c_Bso(zCJM`?F9oVj'
    'u5A{N0~<i#yh^mV#4P8Fp4&Q@=7~#mG3Dcq4Yg#8jvcus*eT?YRq+;Hp6RqSje6iMyuV!BNOz^b3BZMVa9U9BbRZX<_hw^mijt(5'
    'BM`0(Y<pA6gV;4DZFC(_P0UGp6tIR>UMJE=yMmlm5yN3=W^x*rQ<zTF)M|F7!Xo4)1$`Dw*FAAT1qR@0i3Y2+sVbBJVy=j_$(7<>'
    'L^GhM9KT30T~vFYRf!UCkv>#0O`s*JOUK%+6EmoLK-2IXMR3MiGk4gLawd5tbgtr+I&%^=?pUT_;57TSzr|v00xYmtXGF7~!5&TG'
    'sX)F^fQ$semca*9xQ3FZG<;pxqE4KAv{0z1`y`%}v+=2-G_a_quKULVuZ+8qNIn%c#zojMz`v?Oe->t?_PCnsC}x%%&=q0X=F5S7'
    'HTyS?g@4_I(WDcko`nr70L`XYTB@pidwKv3U6gI8Ok7fQ@e-(--U%-wh5v$qCoKY^_&zU^6eTH3hpQN+1L&&3J<3n<{3!cDDHV3q'
    'Ndi`)x7sRG7y0D%d0=pcT3juczNMz)2`SH=QkszIM#O{?aa{^>R5F<u-Hm4E3E{(%oRLvs;&veo%2U&By5IphtI~YPYe$8Xwy_KC'
    'HB}OTC@EkCd0|eadleU@Myw(g6joo@09bjWWi=*>V6zjBg?6QsAns-e@pO{J#Dw6@DjS#w=wiSVHqVa{BfERsZfz&!1_*b10pLVd'
    's`bKvg9vgNHF_0J8RYbn$=G;Z%w!h_IGGd+5r9;B`y7@HCkjjZf(z%R2D`k=nUoeOo)IjFIIxzBC)riO?s$;Xrj=%s6whj8pJyb5'
    'WuuA8H+m&l%9S!nTn9*TXr!m4FK)nLB2?LV1qx-iiXsXW!jP!<$5M-4eb1WuPTBd!DLbK)t5haC#T+TXXCe5=X>xWTlAH+kAD_3Q'
    'aiuiAgbeZY%ZY+4dKh*FppKlLmLt?iabbr-lRL|i*0?G?G!e_Z>XPJ8Ox9r5VJ6Qgq3t4(a-a$s)3$PuPuB?%16yBy_TgSj@)2cZ'
    'xQY&i5G1IvR)esn8?7co;xj`KjSqLv@}fH+M0h~3_1q&sfAyBAq@_eUdDA5CWwF{|R`H*MEE?+QMc1udtxzLuX*uD&OI(a?mIw&e'
    'ib{D03qhrnKu<!xTP%l;;0fO~6>9(*<!J?MW2`i;>2K#|5-S$UO=-?GEB&XH)SyDU5;)KV2oiXWb6m(6wk8kL5d^E#tqc}o`yoQb'
    '%Y;gjNlYy}&@&_AIpckE*<U<ahjxIceH!QvDW5E}F<zVQQ$cqMRjjfdyFM1Vr@Q`5KF^^q&{GTXM5mjiFgs-z9z~#=sooHMLpz~J'
    'SvD7ep%{-Y*&8tP6DlDuD~rQ4e(g-KL^xMQHXj{f_O&1gjwG>lm<Ruoyu8#&AsV4tHF+A~mUKJ%P^u7NMkHrr7t@)5VBY)W1hGm?'
    'RFFvukavOBbyqF`b)#gWZ45Lj87)_WGZ%`ruX2LRnygW#*t!on&`VU<Jvf;xm@zeJk>(D^;Pr)YYK7J52{5Yck$B(0#ihf39KH(h'
    'FdDVD+o=Kl5|Qr+n+Z+;&50Zj^^NYvi19Qji%zewttH@jf@)%BCI$6qld~dP2_^}m#nHBlBP!(wJ`L!#(av8XWIKV5F>PlCnZ)i8'
    '=ktu${AH$^A{|sZ3mHz=v4}iZFO=_<#z<D*v%C6acWH`uf5KEQF|Spp(YEx@w9bMZ!guvDt+)_a&AHtiCBG}7IlPsMGD*pcv+#3A'
    'tP7wh3<tvwac>3aXtb4@UZrAJt@XXe^OW!#S;1h7fCW|>SLkqS2s8V0<4WVj4zTFrvWi`F|MaQZ#Tsru3?^(LTBGy^&c!n=sEPI@'
    '`|^oYND;b8Lf}L*njN;_OZd$4kd*l;L}8>G$OO`9O+gl1U>!!s9g|xLk1vo+b(O#`V%}(kDKqf$X#?h%41`UAH-jT$<SZc3UY%o$'
    'db|BBIk`|4)j5*sZ<(|a4}N2l%W{kv-W9g|s#fxk^FotS@Uzlp!ZuX$g9-CTTNy!bb-7~(k~pw3@DpaKOhPD(w9r|_=NkJ&6zz^o'
    ')-;*Ntx`rWq-LN{737Y9po9wbOY(apd7lYpzjh0Hy`DTX%T)cd6M_Uz?WrO}oDY<Oguoz~e>)|#C<Skd0XQkA{D`?>A{&aCo2-VG'
    'oM-VQey}_W18RB+0HcTMO_osYq#&uZLOAx|8U3h7P9fu_6b~X5#n6y2*RNiCU{lj%3iwW}h}<rrOZ7Fozd~;URl`^oq@9I|n7Npk'
    '-ACs(L~AIf%kgQrH8`7~hYqr=5f}IhTJbJg1KJDn!QBDT0kDl3cg}U6-T1f6dLlVp>rCW3bnoKcr1FllC84z%O2N<uG@56*k%D&|'
    'O5aj{3$Ju3Kp==^NW)82sXgA>)_46Cw6>Ip6TM=W2f<an<i^CI+~Dy4%jR6`(m;&C5@)1(09mXGXtc`b+#W&oiixgaqXtg6pOMFs'
    'w4CT+LUSx+;Zb|t)_2b}`&n7{!a+)wkAHp*pUtUPK7WoLma2+f9jEm9fezPS*T;qOPP!(#52Xm&&k7Rs@|r-IAf~Qp#3Jmkr4YoB'
    'DUvOw*ZES^fl8<k#CV)`F<tA7^i4qhFBObWNI{F~>{|%|YEu)HiU0wdnsFXL$`Hzz>sMzC7#+^KLP9#;QWJ!gBnQ~kiqNh(i9cgI'
    'rX?gsbQx`(ZThf*f)m*D+`~D^gRFhvBSGgUqpd@+1BZAv^+DDNNh+QvDKb3bkry<U*GSbUk;togf{foxJy%E7#1HsaVO6&6;Ue-}'
    '@<<B1iwe7KKd$*|F9a$@#ON8^QPi3Z@udXf4pddzr$Mo~C`E*eqG+|=!N^P5PSfn5bN>PX&)1nOOHn2$rkMbmj8wp$=0Pz<``uQ>'
    'P$-egAV2f`Dyoa+6}8IrgJ-w1T6KMTu~=pn3`Mnrq*lM|hHbYcil*3L?RCqZ2Itz<2s_a{wskR&P4MQMc}kwQ_452BV6PNJjO@RB'
    '3n^|@Dv(YS(V(4FS5at3a?w)jS|q0RSd<oxp(W)F_@q@+<MKI)`c|bnRpGx!MKPvWA?M?ha~jUqx}-(J1dimnVZ;IMlkVGoSr!;('
    'wfrO)1P9e3{6k~;=tZc6^+fe`0?(SlJ#r8;QW#c6hgX@>!Bmfw+W3?P5U>!cYESN7_mVauc1|+yREbMBRhJ5LjW|@bO`$MCZ&r<J'
    '*HFimx`{i=JYx<rjpXO}|4&#R+MHwWdQ^j}th&bb6_Z^H%hod_kwy1!n$|D8q3u+a%mbwkJ6p?ZH@W#-FDst~vjDG`E^v#uNj1Au'
    'S)LPzYN^K+7%LkPyh2)6to%;n3#BWCl@&{<uIfNMUCDYVWSP<YAtl|Z;I%gmO)X#7TJBxJv95i@kW|X24SHPXKE_(b=xL;b61X_T'
    'EG80a8Y)t=1N<(~AV4+q1UQeF$%(V<c&b$A9rQmtgM5oqC#_8gksDj^^|~_P%U4NT4F0FL%GJfOAV6N6E+TNLsU}Cj`Sb+s6wB<D'
    'P(B9tg_%d8+J&N(z2AaYYV{LcAsCw&N?F4Qu4vWE7N7{Q@|dJV#I|GDShjjo8SMj$O)2;$(r09#Z=D7DfDS(+axjJ(_$iHmJQz~m'
    'QGNNCNnOe4<|SZv)naRKOq8SxE%g$8f=a?FMN5bB;W*31$tY$a+X8LeCMki3bDHJM7*(uCSN@W+v(<v|AtX^*=C-0;v}JA8a_}b+'
    'axa)7%X;{P@%FT8Fua;u94AR=;ZbdVG8rV%P7|k?6LU07u8JanBT>|aFY{x|AY+2N9Kfl_)R-paMo@$WrFqt285&h%HeiO#`2e|3'
    '0SH_c#m@$tSSetpn=akei$t6gt{o~~9uNv>`93#^l6g^=Ok`>vv_J0~jY~sb6U)P^e)*IQ+KkP7Yhc)SWiDB~;oB?J$0)NQDJ+v^'
    '-$#r}2ji$-v2&jH&OsD_EdgmH!a{@w)}i{|@axZ)G^JP|!#)YG*ZDInVTWAf0LBUjIiQs!@(b+;T-Ke7_ePTk0nt7wIC4TX(f&#S'
    'rea~Vs&W>bhuvHeCt1xRfTlZ_)02`N3DPOmvsO{sQIYYoWPKRo-p`6Ea*P~xHG}Y}KR#_}Qx6a@p=A|ifU7p8-F?+GRm^jWcp}+i'
    '@nJ0{t$iXC8;NDn+&ehn+aG_-W<%Ngi0M?SY}YABz!)NH!w0-+=Zui*tijqL$f8rxB`~krO2Eakc9aZ5!^wqoK{59$s*|Z^ddeAZ'
    'mj4nDI|Xhch7PIJ3)6u2t18hveGDlH*iuJD=2MWXZ_yaOA{cE@WN}%%h=Ry-9rlD9cAeH$Z~Wl3XPRcre8VfARJMzXKoJK;$qEAN'
    'VV~^kqZS|kxFrvU^u?nd-3luvtEaO@_-DbQRAFO!2?4X&%*6!9zPo*YxCC>+0~uTUv-U}(rb&7rWt%CuG_B1Mc`i|Pw&=eSzPgQb'
    'G94Uxr14NWK&eVdLcm0q<(m0gwNFYlx);k#i8!{BQHrwpI+qa$Y9b|Xdy*3txt`QGw|;uOOo(m}Uz9<BD7!|;_iLleUlHPAU=`Ec'
    'KUe9asO7JKC{wE_b-I_XK-ylu1OONY#!l(OU~&74WM0_XRb{<M=+Kc^Iy+5Nc)&e1YZczuA<X`UmR)F$p#QTt<<u%%)C!T&{SjBL'
    '%Waa_zhJ6s%$ZF!gMzK@KZaCHCt4m%6N8*)<+8}Kq3aC{m>)H&T@K*)NUc+7Bxx2$rIqYLXGe`(saf0PitLvTSThC;1GAM%6;fTe'
    'RVD_=i9Wi1OTZqR%~+R#o^C>W?^2J1MrSz%`~NEnvrahnB(xl-nOs)%vGTwQ0V^bCt0+Ocyg(!_U4@d#7+J`R<1-qlRE*m97<fSM'
    '7RkcxbO;z2k_ubVZ&inyT<Ut6ZYk};A^Gcq-6U2e3U^~8R?MO3N{<O)X0i61H8T`V{Aub^4U(&+4F&F-npqh(T#0L1d8`D7if?FQ'
    'h&1rL0~WoQ&KQ)ywij(BkB;4(L?hUf$@1i8WYT1~V5BPPEFm$d_9ym$5di#RI5w7~A3%H|d&g5?h&Pqc$W*Ww2nHS?IaRI*HWEyW'
    'D5m^wgssjZAJlGOop4>n8Y1JYZoQ}?Q3>|XvYgrIv=QuIkTyco7@c2sa{Y8eJQ`j}r0HDS;N-Mfmufib<-@XqbKY&BQgW_q+Camq'
    'Xp_tHffymvEfKbxA%<o*?MP}@<`KK5HZjC$2`WlcT?ui2o@{HsLn*okO`sO><S<2YV5bQaJWor>_m@(16ITp7<#@14^iXRwl}zOv'
    'bcp55Djn3+s)Q5UPgkJ~h3ZOcff&`gR3GHzRMK{WZ1AVOd7T(uZfJz}p@55b$|BxHMQalz5T#Nxi;6675*qqs)2d~FDL80OA<Z}?'
    'dLFSU$^cA%!#1ISV(x+>k4fXit*j)*?U0Lw%Ly11Kl0`=Da7Gc8V?U5FLhy+*drp5r=a~M7F1rYjF9t0D{}wnu?3r?EoO1X`J*Ba'
    'Nnpci1<SYto-bY<DSk&OcP!2y>rC{h7^Mg{Q)9*5eo8W(t4OX(ip$11+LkKWdq-0?3yNS-RRuA?%+#mMST55=K$Lg@mk!HNC_-pA'
    ')7yk}7(_n#z9oB;{3%&lVXda?uvuXTb`(96Xm0|go(gZh2Ebt*j>Zghy>wFg&RKZ1dv4->MwKTk+qF`Z+BH9OAO~y)rL9EXES_IB'
    '>*ZvkFzq1KMB&RMZL`{mdZ3F3j&^EdzTgC9YnCR`=>w=NniGasYjrl%+#=j+x#UD5p=W8!$gFZ@dvRlQ0SEp&nWeAP5l`4CGTA|p'
    '1PSV!@r*lNP$p5;+>bQXEMrZ3kKX`68hFii52v2DiD_$v$9V-`k#q{BdRfA^c&eXq=dK_7v_D;Y8ptS!8ZR>+>Qa@3*Ypayc8#Ub'
    '&w0n@zS)lpY%X0TKq_*01f*JrM^bC|m&^r+xMuo{^JwPNi-MkkrrJu~Qt&rX)MX9$(#z=Ceg-#8cV%`Vn}1$$vCZ*-$}3(QXKCbe'
    '<ILCq&)p5%`r+>O&30IE>piXY>!Z;q9^T3RMm~VIWcp*PZa>Rw=^Y$pi#<P^Vw}YW!+#sUEZ<R7k?=wV?foGz6}D&1y<&Yy&$WG9'
    'Y9K(jliM)a#--fB=tG<@@}=>Q{|8FI|Lp'
)
_V17_SCHEDULE = json.loads(
    zlib.decompress(base64.b85decode(_V17_SCHEDULE_B85)).decode()
)

# Frozen public-only 53 -> 16 -> 1 tanh pairwise ranker.  The model artifact
# was fitted on chronological train data and selected before the final replay
# holdout.  Canonical compact-JSON SHA-256 690384bcc64e715b1860a89d97657cc96167096678bc3fb39dacb1fca6641d8a.
_V17_MARKET_MODEL_B85 = (
    'c-noOZO<OZk)8jFz|SNx{W9G>`sM`6#)viov;vGIFbG<DY!OS53`uEMYxuwCx~h68QR0TXw#m8gd!}EiPMtbcJ%4}n{M+|0etQ1?'
    '#p_p(K7aK7`K#|AJ%04<^OrCF@-{9$d;jpmn+N{><J*TvpO>fgeEjII5An}uub%(#@b1y)|L4)`R}bI6e*g5p{^9GdzxsyX?|*;u'
    ')j$8?o3DQP_rHAe^&kH9U%vSI>;Lu3wSWHN%dh_DFaP}FpZ@7rpZ)R6|MAPe{`A#XU;gs1|Mtb#-~92*KmPA8zUJC@&wqaQ@6X=6'
    'ef`~!-~MVCA7B13nvXAkG^me1y2tkM)g0n)zL_Ka_~IV$#}_{u_nVi`e|~uTsiFVwlMkc+-6tPS;CG+ibNJl{IgQ_cKWFm0>w7Z4'
    'yZ+I9zJLAW+h^}yy!zCPKJnRyIep@@k7o6W&+mDC;-j3|r+$-j`@|=EcAxm<_w)Pq^}GLcejk7K@%%pi?DzBg`16nE_wh%+n%{4J'
    '^Q-xN{K>E9_wgscwZ~Vl9=?0_f8YQ7^yTZnl{de+_UiT9AApAcc=(PFet7=&UmxB-ee?Fkw+|n_{p*X*etP%phll5{K6Ux|zyJ2?'
    '+vl(TRrmh4S5M#k_?MS2zJ2!Y$2V_Y{xXo)Z{FbhuipP>AM?Yv56{2*`Pt7e9$tR;`=9-PKg^pwd-wfwVD4Afe|rA%$A@1o^Ne3#'
    '`SasPFJAri;qAk#Zvn?g&;Rzfw-0}N{{G>!H_u<Z{kIqI_{-ae{W*Vq{g>x|dGYea`xg)IK70Ms!`sjP`u5@BvsbTQJ^9mr|Kbnd'
    'eD*J2eEH>P?_R!s|Bh#cqk*n}fAsx}@4kC@_3SS%o(s0e)49eq&(+54rC;-Lc{<Cwj_W+vS;kRDdHm#fI>uOMou!r1S1IH1@pM&w'
    'cU(uW>nLr;*J?fMsJ-=mT&*6X=f!fiv!A1^bB=KiFP6y{X1|W};Jb{0J9@*~c;dlB&T&5WYvsJkn(G?J$&LKh!R^+nv$b}udB!tN'
    'et7ZcXlwRWE8jcj>a`cYbd^%u>GzIxu6|4=HYy`+KI2%U)T5QfqsP@+%%pUNT+Vr%lfSkj?#LA8Rjzi<>o`_h?ATt;_phJ5{_*{r'
    'AK$a4PwLZJj<FcY7=`7iomFM^`l#OPEVH)9Ps-E9A3p4(v(oFh9@nR5o2$)Z9ZamTIFGp*V|Nx;?dQ?@^k%EGD6NipjPvYnQGMDj'
    'rf}9W&VJOs8ed~J^DI}NlO36L@_>3CtZBL23})gmN<WTSE9-hx=GFKX8_Lv~>^hG*=8Rzs7Hl%_T1xBHQMlfiL@)K6lUZ`J+Q;IF'
    '$BNCkocK|$lV7sn*Tr|e*D>ch+tD#6cIP<5^jSUEF;|pz#5noJaZN|H${Lty*v?VURr*{97qE-6H>XdniItyi><weiSSBCLiV>Br'
    'MfHwh)`k_Z{m1g!ag@5Qa~-U%A0IPJpbS>|WDT&!ey(j)$JqupKiCp1IL5;>YHgS#=BJ0mMdpXyG^X0BH_40K39DzAW3YBPo!L(u'
    'iF>#j7>Um*I3d=}eJ|!$N<IiXWmkC2<Q8+yxu3*H7%3*lda;3wy&e}k%JqvMjRRA4IT_JaF-qph@UV)!0meG<2rQAU;%_U~gk9s7'
    'UH8QcdX5#-y0GFg@UAkCT63}obL8(QwpK4)Zp1B)!Q)T<)Abo`H@*y99H)+bUYIENi7&H&tpFa3eX%-ct{9$B%REo~hNWSu`@7d%'
    'g+bs;e0AS<f+g$?)>e4*3<P1s)73h#Qf%PpE1%ZO#pv`tzIV31pH@f1ffi<iCtlnuFXNZjIEz@->%4LhtcjjkFh;DW6xKMbQ{ygT'
    '$HJ-S@(z3p<Ge;=er?V$L@bH*odApru}ul^#eV?S*}y7*EC7^+#iP3WpTjSVMU(|(s}pO1V|WQWRYheCsibkRd(4yNWAmBap3f4p'
    '&mr7kBhII=az`H+K*zw+2>746z@H1}ZTu}p!&>yN87+5kyQHoMG}a67XH|2Mhc3Zb#s%!N9O+e9UDGtNZM?`m@>_-FWTP;wbL2u|'
    'GGOIZ8mP>F#TP64(9Vk2*OSlEPZ<ciD9$uDMAO6L0K&#16~0%)5KfjGsAJY3g4h?Hiv{rqlV$G{`VZ|813N%_hwWxHAYcLI-Dq;Q'
    'm>*cXxWXkI8WWAN;POU0_KUHX?KCF~yVy@Ql#MCr_Lw@LBQ~%&gxj>V0y+#=_cDB9osN>tAV@g!%EM=g*=W1`v=CUvVmI7EtmI%x'
    'K|r>e0JJTV*v`rUVSo{P9y159S=in+y`%S7q2e>J=gM9b;3t=(V?4&i1?pGK>2@|Ia}t?{k*#AwPp+p8kX~*%Pn)AWJUMbz#t0mo'
    '`ikh0;6NxI3#((mK=k%NR#J1pD2WilcuoKp>nsXEXkjKXSVDtS03J?2DJH=3urTf=z&r4v8;K9TcU=+q){R1J3qchZ!Z%t(>F^AO'
    '#LBY{Lq1e)+|^2mIaoQ^v=<g=oFWd33W3_FDQpn=3hPV&&8-#*7uK#z^ZUNT3v0~IV5qQ|7MQ@Tc|IYw6W>?Hp@UHx?{K&)?8l2e'
    '=oRK|BmpgA@nj53Bf)$z(slouwZLm}X5cdx)Tm1Q!O|Gc+}wy?03VPM%*wc>iSsuCEQVvGkH{jr;g^R&Y}O6_O-hBEEqsI}^lMrn'
    'I{|32Jhdc{HG!6>xeT&!jEG7Vr-b@xGcb&r@k>flgzamxmPc+R7P#aMS2qaumdFbC0}a^W8msFr@DRqwE;8|>xG{c)n_&{52J|tR'
    '0Oo(mcIBoCWQ6fccF<ilVpjYXpd$9L|G*<u@^OrLxpzz(W(JIMjo9Fw*j_@`xLQJYWizofwr*@lHd+APhi1NrWWjU0>&#*=0f6;M'
    'Fwx%m<btxbB;v&|dWop43)_@SC`4j4^7CU|$A%IC3Q|8EEABKS=-@4w5Imt@(s8kk5OlG3_`Kkg9xG@sX(NdNKL}HjZ}QCsn`)x9'
    '7At@d6$>~3l3j5L<RmBHR&3~r9Y+EoG2<B#j53Bv)(s}poFYvLOf2$QAWa-v+j~G*oy3^Vf_@-9j1>|L%mN}gK>VL&Hl7F0r32t~'
    '5*F;m-AJGSj7bylTH*x!Y9>9x``B^$l8DvQx+6_6AhI9^-3%ps72v`n2{3`j?f?S<O(qoLN&_Pok<Y_5%~)zuP3(>Z#RBlswRe@v'
    'm_C7?ahn5#5%UwC%tYBVJTPI4;DBus<qu<GUh81VX+DEh2{pMu#7DdgbSegjTnm<;XF#gpY2-JfZzP3Eo-l$Tz~{7!m?sp#AeRv!'
    'f8Uufa}?(XtTdtyQ<f$Y%h)=I#Rslg<x#}O22bBymsJ!La0p{(of|X>W4IXHm?z25Hr2$91jtGn#qB;&O|~1JLQIj7?O<Y(gTWWQ'
    'PZ-@y6Y7M);s*f;fi1E}0mMK80K!K4%{2XSAo`(U=6A_(nS+5B>~EvrjF}9=O>KeGb2CkL`ZAbcm-WiI;UVO5;{rR#pl-&^l-MA0'
    'C;1=%zE_9v%4JY|8AGqSd1lpss+k<{oG#C<b!h?n2#jDoVMj1d2Gfm=!=xfyz6Vhx+5jFqWpHOq8E|DraGr5h#{<M67Wlx-cnFB<'
    '5hf8#XoZ7UFA<@zA8Bokc`^hC1I;1K*NmH3u6BkWfU1B@0Q)j+z<`3ouDZ=17}Oz}{5`{VV>m1f_`p-)XET`)1p3gaG=SuPgxOO#'
    'f-+RmFDVnT6*q~%4!-I1!~`%Q0wic~ZMt5ngp#1ZJ&kq=bX){riSJ4gNhryOKrUw;Id+C{$Z`vqWhdLr#bH~UL`@q%t{u5Xz#(V@'
    'tj$1-#t%-`p~*m7(oCa#k=(MVBNxozZDP-MRsS~@+u-fGnuc9iTn0i62Ka;jVX`HnN)r`WL?wpn4#J0Es2`pVJCZN9ktvC9FhOiP'
    'zDL9)t%p?$4yUHbzMewBg4dYj^vy3>Qhu(|fV1OonR=Vu>4B$2IzSibh+I~WVfvf!nP^Id2|fU3xnW}Z<-@olk4PuSu1q>uIe_X;'
    'D3+TbJ~DaY+x8@QI1}zp6Wi6B13jB@OVR^9J~w=7uR{hPAe7XEOhK29z#@a-L9Y+x0{5}#PuFJbQ^;NNZnM2RA1*ElM2rzDX=P5|'
    'M7f^1?Ab%j5($W69ixOWO#=B|Qx@rztYYaS#G#0o%OD077^(u9{hY=(Zpj`uQ?ty`Ao~O%QcR5L+Nw!0F~C>}h0N{B)g$6TzKqvk'
    '=h_(+)<(ooO_)cbBLSYs1YTeZYDp2_$Q?&;8QD_ZstG$U6DyH1K_K#CEcu~^MTjqJGw`MzOPIjOvdK30Q3%Ao@%=OO8IkZb;ILyb'
    'w(tR$@Cb5+)y>=sZp2a-{#;c9xzitFlH}WfBsccg1a5OBsVNAM9@G^vyc<A$8kv<OrAi>0iOPo{1_l7rd>LP0d&A%XQPUF>e{DHZ'
    'PBxzGiGYBgVnc~m#0!F=L<i?|TVXe<fQ02LiHcOL!1$XPEDNR{o6C(QOx3y$Gi|92uszB|>V9Hg>HQWgU|!5oqy<T4uI!~&fwv4+'
    '?I<7A7y~g;P7SbbyI{k50H%M@xss~u`Q$*$DO}0yWT{dvg5UC)iI$7(vlvC$Z$1n#s^%*O5`$r|Xjv3vH3kwm#<o2bMk9{YKbf%Z'
    's^Y{Vo3*VVo&;0ANC?Jd6HA~v;wHQW$SY&31FBkpxNd*MNEE?{X+0#jGo|OyC-9_8qdkI?8{SM(x<QGAbW<%xS|p^bkNI3IETEt='
    'td1aqPrv4uT{|@QnpXyGH94>j5{qw=Pi(G5_+yYuyi@QFu#guMM!W&G5wYbkBB}QTRN#suFXBf=dFZw5JfTvRqDP=u=JJ~~^KiCW'
    'f`GXuA#J8?#$JUIJUXo^u0Z(NTd3*Ui8Lg)TP(obpr;AdHAGDnjHQa5q=yiF6^#_M#9sQQc`6M>U5Jcw)vZOD*aL@BTnuhAw;C*$'
    'SV<T+FPTZZL28_AsmrMJZ>CAQe4Y^#!NztAhwALiTwq+7{pL~;6Zx{L0n^(qAfF`f76Bj<H%DuRN9NIYW+Tn4>Y?gsX5ZINfEC_M'
    'f_juVw)l)YFJ1p!a%1e`9v4so#KG!W`gT)ChA5{pxdkI{-y})V{7M%-`{skiG#08uHiM<J5J^-qZ@*S&6m#yG0o5#T=oDi9&7r}q'
    '!EH#1uaF93V8#cC@{-Dr59(1s@f5-i32aMXW4j5D{2yks^{k*R3WYMWo$Imz>R{b8;%vz*fGSeu(-gx$N8foC3vUjf3-`#f`Jw2;'
    'B#T1h*vMp987`j5_jbWU85!big5OM;!Y@PJ(_4r2i4rJuD18F9VD6PF2OhD4Cn^Uq%47;e?%4w8m%Qp$8kJVA0PADz{1R4eXu>C|'
    '+47bamcjN?ytwX^h#RV|c2ZZ!UC9+{Y7iGOG~(WEdzVFz8q$VHwq-!VjkY82+e-;VN+K^F!z6aT%#~@l6@gdQEp#mIsBL0^ElEzb'
    'L=~b9sgyF<Hqa@rS8CIkX5N&2ma>?Ew+O*O{IngQM_B}>xFt}-P?45O1lqcpvUnH?wlYHI7C<JX07n9@?yN7?r%M!qy+v-pR-E$4'
    'ATD(i_am}-;rfsXj9a_ymu!RNn6Q^Wx0nKxA`pOf=e03cx`P$7ewRkKc_*0evVhJQ-{z78%V8Y}mjzGGWDB?ye$_3VS}RS6V}F%<'
    'RLcUR(qOq;l{N`6>)25<lqRz#An^v+3s{+|S<UElW3kMbC}cQd*qKFD1CXVHC4J7c_As5TQPp5UdZwBfyGz6lo7<Mm>PiW8wwS}k'
    'u~j5lD;&H=iXsb&M?vCyWO=6Pn?*-i%e+!Tc9fLYXQbp%KLH1}n0h9z$WH1(Gvm!AQ<uWR#;pJ4wlOs89#;8yT!QVAzZ$H#sj~Zi'
    'wy_plW;)h}lC4|QRfd~tARAEvkd)jNM;O@VOI2Axp-2&QGto%vTFVMn$rMYvafJUDH)~63q4x+J#{6CSm{BXziVjnq4V$L?_=}YP'
    '+9|D!r3<|A=q&|Ef<Vb0deW8U3jjJKdn*lWxd}>CzPqkDRfC)QRXDb#7X#=}eOws}fIGfvU0*_5^Oh{8)We~xp@tl2$T4p#KCb76'
    'cO`v-L$|20GoKQLcuNAzc9_6^7&^2mh`w~tUzd<4MbBx0?iR(37Gxydlsdi$m+)t@4ogNGCM-cOxk(W0ZI_o8XQnIHED5{PWBIx+'
    'p_-B9k1G;q^D{zkE~I@7#x_MkR&AHr2tOi%AJ;9dnq5vIrpVIDR{n`_hpH(iXk}%GpcCKk)(*f4?4WL0iQ&y~$Z1(3LQ7Q@n^uIb'
    'y*mV@Gx($KIx@0o)KOO_i8NGZ3M~Pj-0axtW|(hd5@2DqZnwQhPQ}BOQmtWcQi-W`%b6^gW>9l2ResGy2wA}4EO$j6v|;76Vx`wz'
    'PzUzlZ!^{^$IdbUL2PhWx!p2)QK4k%EygB8F%zxwvZ)eP?sB@s2GdgF%?LD?^)EoI9Z|8p%jtvRPh)w%)oe^w+ai8rw8ySv0=`8m'
    ')U8F`@hi%*yRbX1)O*1hsReiATg29TiI_?e3fhvxZi&;?0HHA0Gx5c(!Hb*OT~VCgF7CEKmo972RVz!}!gmfS8dgbg-!xf5egqxD'
    'htzt1mHfM&M$nS#GF(Qzs42IQRn0pJg;sv48xeD$OM9u8>{Ove`m*)CL%^+3*oNBVe&VQ8?YXOXo%NPToAiaab4n;J32p=#W}YK+'
    'ww)7(7JkUc5BY0foBo0nE1Foa7R_&qx0=D_Rq<<A*IKx;v=k(`lT|AZ@&`$ku9G#5NINw~f{on+$>^FpGfjobZ&GE9Twd3f#>+Q~'
    'J7#pZ#;w6cnoZ7OQC^!T+3Z_Rv~8lor>ropTTZteXCnz2+U!{<4!5qBBBp75=9`9^A}`kvQ(cqDnY3G}Z&_6A%J((OIsr`KBQCb0'
    'M=E<6hoT}ZU&iUM(QuIxdfBveWgHG6^O6eLDuF#8bBnwNM235sm(C=dHK3-kCWf7sO&Qj5acqL3qJ0#Ttxq17)$M3$qDpY@?ig&K'
    'QNi8CkTbe4LYke%#hqo0FcPcWaMPCFX_>Ftykk>M-*u%Z9ue&Yk`XJ>Eh%sZ0?sgK#SB-pH31&n^4c1)v6p{L>>WoqQ^hIP)vVn6'
    '<Ba-vQn!Zea`mhSs<7+qm=envBaw$%ELuAGb<6TsNY!C>O`$h2b1K~GH^P3~MFcw);31b3f&BW~V&Ibck)p|sO12n8Rb8>-FreR*'
    'vW2|Ktn9tn{11;|f;It}c^@hEGA5uJlj`c^J04rbk)W%blGa@*PrBJQju=BT^s+FgT!R=TZzx&jH-%+aq>oBy$$=shVaqQCkdZ~L'
    'Wu3U2J&&@39M&pB(+%{>O>v14hZMoaFUtzMmR~LFNCUuyE1Qt29GOXXJE0&7X@I83hvdRUJ{j^AB^GaCu_<s#<>s=5X-2bO>bFp='
    'lzL$UZ4-I(Xd@4RWs^cQkFXI-x_-sz6YVX}URG^-M1|lCmb(cY!|bDFyO6zr+>ieb`6sh;H}*0#fShed54@WhPpfeP{%(NesTfFA'
    'Dz##q9YImU77ln=W)I5k*2;9nRg$lYa`?6;W@ppdk;gP-IY_zN&gOt);+DWTHNwZrW}3TIK}OP>8>sjbZFBn2-MuBdu{kk@jT>&^'
    '%*c3JhYe*p)*sVkI2JwFoxl#u#4DQ5EP?LMOIt!^JptAU1|bDm0}HYk%@R@=4LUj)Y%)qJZ|j0r$dPT8a&9;oV-OxdvJqmCmW2Z<'
    'J~qGkIPGm&F*sDFTR_Air;!E_d8G+07tqYJ+PpHYe(s8e1sNj4CHF76G{qPqMU)`iTUSi;ml~GJ%IGQeTE~fwk)~+dvQ9Pz$+i8j'
    'SW=dPG%f&q8nA2bF`j@MnmTZk<9@I%>b@CrkKMK*U58&Ewy{Kt72oU%ath%k`e|Y%Y~ZT?+TwrP;mYEuYRaX8v?V5QX4|0}4Mvum'
    'h2z)?zz4E`3}R9sRXXvGP1=SCC;|f|s$N8{csN^QQ2->>E_S;OnWgZ_Rwy{$brN$>sqBzE?h>~(QPMtEl8<c<x=Xnsy%F8o83m{a'
    'QtY20Qw5;kO_oz}(VB^wL{<=vNZ(l|+xViHG1{^&SrWtz7q+tWI4Py#m|`AJ+V+eClTtGVrg`jgN{PbQW&U*}$0Uj`<+f-OU^q#l'
    '4{b~mM*Kf^m&4(n*<oO_UN_5may>omri^Na^KRKZRq#0w&r$`yGF-*=Wo5$diNnGTVLYx@3tT1}nvBBjd3Rtrb|y|j=VIBs2O6xW'
    '=5$twSXF@`zkNQ9CNSsPqH!fI=RH7Gt02`_@47vH((OBB{|FIN5xnhZuBR-H60-FkcOfnh6GTO<qBL9DM*}ny6^8B98WO(iX$zs*'
    'jxLDE^<jWLdN)n@tq|tq`_ous!s=m{)cEMu)>jizqTO+$fL%<AHKzje+Hzb-CnJ<sEL$=HHDN#O05u;=tjNv+Xpu&dEr+Ut#jZO$'
    '86sLXP583$j-U5khqJ*&VH!AFfM~?OXjpc=Y~8E@M>fSA;e+_B?b^Ft(WLv-C)e(TF|M-+YTwtfslGjd(jq7XTs5PyC2!>~^><r}'
    'vm&%C*)|nJpknuUm}sc=gT6ASKrM~ezR93}*N8~Y)=XRFWNKy}I~%mm3a$wp0@^)+Cs1pK%rc`wVV|!J=%&jU)#eM`mXXt3#Man$'
    '%e_S`wDf0r&$W(fGc?lVCu$aQ@fn!Rq8ce{gg@5wFv|_8I*ozxMfJfEd1hd@w30hTv7y~G8CKwux~Hb_KEY+_7N(tAP2s*-OtVjV'
    'Xk$m7S(wpsa@wq#<wpAi49YB6u<_k*<Se*(pt>AAz;ie{>m=6qH@&hyQuQTzbF<N~p0Pzs3%(n{Z9rirdS;V!-I=5GZz(TX(>{{g'
    '$;VEaV2Gf-Zv;~mH|Q<%mCP4TF-*ExJ*+V||8%w4WVU#WY_T(`D^WUT3~WWY8F97IOFjG8izdAxDA_#1ULCsva;Y0a8LWHVRUp#T'
    'rLp3<iph1i(=N|FGz-)q)$T$7$^wX7rR%Oc+u&%`yq7SA<ZV5~{i;yC?#@YLLG&|QKD_&O4Isc^Gj}5PjGGp@!-|)BM0C)>yR1c='
    '#^ttUW4H}KyR1FSny%fH1vghvIu}9vZZ~05p_(%~0{d=O?xsVgJ60Qd7*^L$iyaWByy@KSCe3{%!q|!&83O}U@31y)AkJ=cS+WHP'
    'YcGB59?j_CYM%cX`bc{9vhrmfs7#b`)@}-rIhwYr4NF_dl9)!nz1x&?0k@^O%Zgx4^X+PmBMMM3x<oCx1ex5a3FtfFa#6GfkFQLG'
    'x5N{L8sQ)gZES{cE(`5L%HBe8u*lPGudxrN%n(d%1yp`+b{*g1byVV5e7BTM&w35HoJkQR#5#JwIjE?#Vis@c-}YqKZT1}4caiCW'
    ')i#UrqJy+DRuO--YDG4B+CN!S8nr1-0I&yjEAdN-+KQ-c-$$mk#YA3SR_&3G!6<fi&CbI^N48m7SS!jNs#uAZQ}o3FlJ>TphfDMo'
    'f|T@nVuhWVY}9aBZB{wA{HhWo3mp>QYj>PhbINX&Y;CBHLtwjW(1z2tQ*^t%OkIm=blqhVBYPO5omaalnIFkxN7Rfn;F(gW(C9^O'
    'AnRD3N*#`o`kU?PkBgAf<%JQ>Ql?eBt3uzfVP#Lo`Y&E)D#bKi#;8PScG^$hyj=~!dXJ3)d%=t$69=%BrOmWC(?-kJLydQ9_b{+%'
    's2_21VHXqZDvi@P)gH|mXXQ#%PM3)eW;Vx_c`IKr_Z#*Y?4~yriOZncj15U>?YOd)iu>7Q<_*v2|C*O^`AV8d6t<DNM|0{{d*MUs'
    '@V2kGw7c<q1gpNJa>n>fB0^iKHmfeh_~KBk_*nT;(%=fe5ezMRkyMN<N!UC%g1Xp~=SG@j2TixM)w3U=L}}W_<2Xf=t~f22cVpPH'
    'c{6iZc>hvajuPlC`q?M17NN2boQ&Wj9gD76Tjl~jE(>S&-bMXl=K|)(d?R}*Yx=7#x-Kwt?paOf%A_R<+#p&WL5ZMbT5-nNE)Ohb'
    'CwVW~#y~7w8N3KZ_HZad&&ZcEdRa|oJ|QaELmpD{u-|4zjAR&jN9#W%OJ)W2ZZT2|knkztmZ;X?tzF}hPI-zZEUQG%zbA%lm9JOo'
    'lEWr7r{@~&N`^r~1x&NqRbldb&s=tE3MPQL6v0DBaL03;**0#klEM<%L))FmSE!9E5O^j0H1TXmb&B?RpX5;rjV#ezt7W-T-|$qU'
    'txOg<g44`s3N`zmGW=cHFG>{{r$<0a9_q3RDD0jt)9l%t*bR?ZoyMGn2=#7zB<+#KpCmqZKU@8u(X@V~%%yK!+3JfyY?|P~LpY{t'
    'jP{gf%r0Jb1n$oKL`@vva0g77X11coVVk8QDxYXq5<`}n^hVWz0JO`qawwl2eGX$20w*J<w}NmT>%%fy-(^7EZiTr|!zz>Y(8w$d'
    '<T~n&kqPRl=e>?{q01&SyMt@a+)n#z1kFvXk^#>sG@7<#tNx6BU?9zYi#)O5I$qJ+I6R+~H^U~hSrK}ba<8Jt@f@UO<;;|=SjMrD'
    'W;v$xdl@AzhuFpzwtE@v+;-xC`Mr!`+R|0ePwjg8n5Ly5MRhywb&O0+jX>z#UH-P}A2rU-v=u(R^pp}^W5$8qSK2J0DtU*#myV7T'
    'c<`;o-UjjTNK-V*T7|sMomZKNG`mjD^PKx|f=BJ9b2d%cC8!Z|wHL@^5xY`m=dJ!>6KWh*i69_|6+4B7J)OIrd<6dqLqD^0Y*T^3'
    'tA#Hlmzh>)50zk}eKe?b!{;T%xGZyoo3|*zcqoj;YRNC7(Zx90LPgvj?Frl~D$hkRFuRS7{4V(!c-^w+b=Yj>$wFJJcB{`4Arz58'
    'Y+ID#jEzUmius4<V}ialz5Qb4M>JvW!%)%KYhk-vsLA}ux}Fuy?kTt6z@{;5c}Q(hgUHn`k>g;H_QT(z;ptIgHdXK3=TPLEk>FQ-'
    'a+li7;Dhp3Q`Y2k7<=?5+qVG&Namp?G}Y5X#LqLNW&taDiuAtS5mcglWNxfhy}R^GBBEn)dCa*){i%kOZHJTDOd$WM7w~X-(8%MF'
    'ry5?!LUzQjtfbU9j_8<Jj&KWkDao5PGR-Ga22B^ERsOXTGYb!PT3KnU(dn^MQU2ml2)#$KKFmgkpToh}E|CwB^}!cB^Lp(Y8g`Ze'
    '#0Zon96Jz_K6r$`XEoeoe8W`S^X6^KYcf9jXQG#jo80aDQT?=8QfWS!BsgNrng$Tv-x3Cui%{gR4~IBB?XSo)jPfm`p9I@f#Y08a'
    '`fF4jWG-!`CX(wQ;7jf8mgAduh?TSz51-9)H+|TSWn-8*P!$J*1=#hhs5qlX&|F1rQ-SrgO*yo16@Ht~1>|E>^r{$L<IMet(*v0='
    'lpwZ7x!v|-<u^`?``vk_f~$a0LyD<)^JLV!Y%etz+`UOh6la@qh(``H@+o0hAzk*muN+L&o=z)rJ&qLRHT&Q+W(?ETxG3@2qv`5I'
    'KgE?)VcGC%J8H<vA>r74S?r>>vOex3jl-kjcB<7|(C}plS9r65^VlUC!i%lfm}twMq^2hc4i<NJm84h@^DvP`W!!vs=8MvXk2tJ<'
    '?gJH;fbc<CwJ{*h{7M5oS`$r)Ss%1+;%VAtmF~=w)6rKmqEp9nDH+V%M?;*ab}i$_(zKdt5}RUV@gYUnXyR)-GePzAkWKBje|wU6'
    'w=&s~GgR)J)sZDpd%M<%GWC)5+2va|;uu>LWcG>cQKIgo<WeP_#dbcA5~+OK23mKBiFQfP0gmH7Y2Ts}Q#|;K6J5KGFPfDZ-3zYz'
    ')|6P*=!@!>$j803@*A5weO|7jt*1*V9(Wk{k@RIR42X_<uU%l5#u{oY5l0rCrx=F8P5f6N9m$qWG!!9iy9xv2<6$GAmD~^gaaNSn'
    '*v)|UidmT*9^j4ELTKuy7-4TVX}R8cPBlc%5gz1Q64LizKca(8x4Xp@>S6(Ec>;Z(>%EV``~R!wGY&iFj(DMpszoT9(y(H?zG~^b'
    'c2YjGwCTGNtQDm*+YL~ox7*WvNn=h?%*5L>S9ey#-Hje1xr*-YppC=ThBjXh;O-G0tmJtfgC$IU!A!&_ygqwQ_qlU6#r|~__EbUT'
    'iB?^rKAr9u-M#opoDJNE?TLyWC9+jul+Es=8{M`NL*TB|&TOEVo@XrkxTxpvtS5F4ezn~J>H)G94M4U#=3`-9Q<7_@Kl41lA;X|G'
    '<A7!V^Pe9-$~U;|SK}y;Sz;@mXuIOf@;@Fw+Alha_mq8jS<{C%Gkx?Lru&^s`GTeWHl_XQq+h)!>CbV~_YdE`fB5d%uWui3@$ut#'
    '-+uq_!}Di9J-mIF?^XM$%qRKUq9^yOfu8*E^39XNo!`BG{_4BuZ@+u-ANwtAe}DAX7w@0Fe)aO_N1wlc`{TpoNAVV=&!Zn%0I?J0'
    'JmOSvy@go0lMVSE)JWygLo0FMvRQM=bbH1dUChhROoK}t#~o1$^hFYxW5!Q>H%kkH*}Y)e=(H3FBgl|?9&vchMAQT=&oS5v5%1kN'
    '!x-)(w1P0nDI=IOdld-M;%kd{LCliff3iekG|a#r%<PDemMFfKjJ7GaTIiR0ZUenf&RZ0))$I^3(&VOjo+s+g?32y5Cf`_~@I92n'
    'vOlC)ED(4l?j?`<C<%Ks*`{lIV0UY_Y1Q<YjXLb^)v`m>f^4+jW?z`eqZMW~5T>@@jc~?WQ7%>L@*(kQhbe`@B08COPpP@F6$hYS'
    'o4()Z^4tC1$&ZbmphTpNclVDe!(|`eiju%DmAg8lj1gVV_ZuS48S!@+8%y6OSnSiXBW!x~DwfI@p>PY&TKMjR550MwgpE4BtF@x+'
    '=b<CXx?0J&9-iz@C9%=$#DZ!H3DJrx+L~S0j6_Vds<HCV67TW2j}sg6Z5#4j$(E;`lLz{mG&=gecIDr9O`J-3zRe;ccnHp?2V9GX'
    '62cKoi3+yh`)GXqlu+tPM9*j2e0zj7*k9%=X;k+{Fy!A~NwT-M5i|3JO3T(X>$Z?g-v*@^5nZ^xbjoAq#r_v1*C;$nGnGw`D9>8c'
    'upM@b2Nu=i46w(H@{pOJYDw9`RK7bQN;rBylkg};fYS4q4}l_ff%yiD<>_(BT9z+7>=Cb0S~E3l-y9+p#jcJlE!q8}<qIV%-sh0V'
    'Ayje|b}-S%x`#C(dmFo>k74S<HXfJ*N<kvy(_d{0vWgO;u;16#zG;mlDF8gwBX9<xW@R<dO?G10)l{k^jhb<$^9)w4zXqBtvbCXb'
    '4zteG$I7uXfiFcmO6ZF{-P@shtS)(|S(K+TdYm6kO=Wrj#y*h4&CMhKSbeo!Ov}xt7R5eGdcNCXCuYs#V=*e2)!BDc9xjor%d$lb'
    '#G;n+x92{i*T=irxYku)6t{MFq6np~;kn@ykZg00H*5Glkd*`e#GW?K&4{s=KQ37YKU`ad?K8UIQvN<7qa}Gal}xffBL*m5jcl^C'
    '$;@-&9$MVzFRiO+4Lrhz$zawV7*RIy4Q@7>YTL0DWVoK#vQko0%9`2jaON4t>;zeHfP2QN@tR$<-PWMFf|#pC@5S?wqxoj4cad%v'
    'kHCgyIFy*AzJWKMePyC9aCxjTO8=++83#8GUquq{;L(RY%kLY>tmpYoj89#1K|GvkJvgc{-pe%6w^nG0La~?5Gs?~O&O7yZ*_^LZ'
    'DB@(X^*dg}#{A6FRj+#ZM>ENm9iNittYivYb`wo2I9aKWe*Y@{Uv`B4`5*uIf05op`T'
)
_V17_MARKET_MODEL = json.loads(
    zlib.decompress(base64.b85decode(_V17_MARKET_MODEL_B85)).decode()
)

# BEGIN GENERATED V18 RUNTIME
_V18_RUNTIME_B85 = (
    'c-ri}Yp*27jV=0D4*K(UBi|2y>#_E+g^?{olJCa31EC>XJ$Hn$EkTxnYYgVUA2PFcJu(=Ji;-'
    'EoyFEH{B$sMeRYouv3~~jz_+PL7`29COe*5;1-+lA9w}1S{+rPd0=BHP`{I6G){_ESneD~9j|9Jc7Z+`sG?_d4$7uD-#8i%Q=#-'
    '{7*ajNErSAYKdpT2wd*SFukd;8<x|M>1NKm7P^`QTg+&0LLbJ&j%6_YbfB^37lW{>`7?{_y8-'
    '{_*a|SHDa?QOifZ|Bvr}vi|hn53lNUIMwSa`}n@Drlud7b{zUye#&^h@saiU_3LV$hPr8{X&9zj{^P%804FKk2<Y^UWpt<4=TCn7'
    '-QRxt=C6Nx_x5i;{rK*$|KCslaT@UY*MI;1$N!YSy!+wRFNd~ycx8TCr1tLr|KZ(_Km9G8{C~ar=1)I;_x)exPyhB`um1g;AOE7W'
    '`nOlVfAi&+ufBeG^&j8-^;`Yh*E;hb|MQ&;`QQHS)o(ui;!*y3_2OUt@PFRE`Q4|#ee;=2<5!<Qy?Ush-'
    '~RIP@e_ITw~wEF{<~KX!{+U$-'
    '#otg>h$)PkDq;Zd8<y}YOPOw{ofDQBmb8_eEQiZZ)N;n{b3o~c?$pb>gn;zua?pN?(;9c`pv7aZzrXL`u!K5fAWW4eYF`^V3wz^_'
    '1jVX{?lLm>mPo9^{uO6oksI;{c0U5F~BA_uI&Imd-JQu(_e3%zLnQOqxkYSkB`5%#_=qO%QvqE^X2D%I6XQY*5(~zRPAN)Hlz68x'
    '>`3+zFnZ^Rt}GEPKrYdq$V`={K&Jct_Sq1H_39G-&$W#>brU7^N*S8Cpz-Gug+Wb_SvgJxPGMn%eOiw(tGpO;}`DldMb8nPqK=>3'
    'e^_r*$`ax%^Jt;`ke=0=$WftR}TV0HtmvO74rhx4tX=MlLRcizy1|#p-wZPR%-'
    'LqlekTtJ1@ogMNA{SJ$91z`uCeLoEFJik?a}){ePC3x#$;xUcF*%33KYDB2ERnc>(4Eq_*ADjPN!;S2pETt5SiVm(-N^`U)*!BE-'
    '~U#E*i`@?&^`U-'
    'm^J*Yj9_fPJA|)Z#3v?Ql%t@zZUHQ^+vMXsd)2b~Bgsozyz>s@f5HX{lo}Pi*xAZWOyn8!z5H!?62o9G3V2*b}_?I-'
    'h^`+2gOidi%d0zxe9Y&p!QsZ&V`W%eQ;SWSm--@9w%v?93ez9QMR357T?nc}#mJX+LOVU0qM{R_VHsxBJN#pZ~t#Syv-'
    'YA82fvi&0%eYTMwlK2IGnrN$EroUD>5W)BcXbDk|Dp3>%U*fl5XiRq-Swm&}^;F!hcQnCz1BR=gxJ9tTlj6*0+iMMEzIho6Y&?DV'
    'nzBo+>|1cbKK>o~-FF&{W+Xph@VvQ;J<+BL($Rd?7^Rwouu)|q1eyus)Rij-'
    '(5^=esBgaB8^L<^q01$d_U(b()4)^tZ@T17tGbA2>C^k>5J@BIl5rE#659BcuK?ivrHrhS<5In@5<J?>TELP|hUc!b{OgwH`hup&'
    'RGc6};{bR^2yhz2E&GsrHeg+{(I*q!}OtwaO{MvOO&kwl7T&WzbCzZO7hQ}0w{KfQr@p<v14sp(qpSsI8JFmgf4)7)V`*?r2#FwP'
    'KyJNk$8yC%S=S=M4!NpHr9HOiL2fK0S)ws@-aDem|@%6R?>%Tq}>Z82Bu&QSKLE`{|PKdd6rXmpTaPJV|N5q{%e-M;-'
    'CwCFv2YqzjB822hZYCkZg51SOsscc){V@YK5V=Y(?gqLdD)jjd`CZQE<E+$it{@3}{p=wgfdAE&pY~oM!CgMdA#mWy5$UqtZ~a17'
    'y#+$OziSEe$?onzB9R>7H^ODOAAb0oH(&fe-s^)9Im>kUG0ZJU-'
    '`aV}M5bbHa_8Ktq6D4Pfyw9PSb*yJ>6f?8gUia|%dft8^Ykx|UwrX@()3Y2SJ8?To`-'
    't%gv>p=@xknP3z8VEJ&<~}F(pMgV%NWPr3jB-=2EL%&A@JR=J&R93QP_pN`l9x?--JZ$uw1WoG%Jfab4dV0@p@F<a)0tO(6SwA^('
    '_q!oEqM`{iRM0i66a0m<-QN$0!S2UKU@IxKvNO)Z-z8}+20f2hu&pdY~Px>vaKA*4wjQ{8hvr@Fn9Q=J~@FX&RIB~V3(dYNR-dpR'
    '>qRH?OBMi+ayt(vn?yyL>1^tqSHEGKWz^*hLT!`KBDbm%!%v77&*3E{{gU;1^>bT0a56v`bmp|gtRQMgddX}G$E4;`IGL2b}=;mA'
    '#~9G7?*65=^>JFgRy9Jm+ShmOY%kRM%Hpod2$XgKfS`pTFenIqBw_-tS9XbvnrDkJmdfB)Ew(iBxb9z!+?>)(gtJMKkd%>q>lZM;'
    '_*&@Nyy5;BxJmX1?v9GH7OCj*`fIJ!h}EvO*Ab~~)1dHD2W&H!WjPsxyA!C%DTF#s#qN#Ail0n7ltuIE8Hbfo9u=3|l&5a4U!?{3'
    'Ig;AnD_1*OM4$Z6zy=JFznq=|}OVe-ue4YSX*&p-'
    'bR9chtWWW*s^ge;YW1KJo*2Du1^$A2fb1(++c8CWRBK&(5FDx;ux{(#(_K*{8lTBMLeK9!&p`j6D09iEhg9-'
    'i}AW44>GrP`J6V>~%cXbC=<OTaj`!mHhr#&9<${WPY}-'
    'u&*9BRqBl<rU=zd(~>!m%m+(r7K<8>R9}pguHnP^t%&UvV2k~UO6&G@N-'
    'FmVEpb7BA*9sN~vVsYSw8%2%008!oQcW9YJOVBR(+GUb7!7bDmKq4bq*m_=LHke}LHqo8*!5gOKCbN^wf!t<u89RLbY089m-'
    'lmtnRL08oj0iJ(+?-'
    '*hhXO%alSf|$5aBXmvH&VD4aqKiaXlEVYTj<YhR%ugZJtIMd_41j^ko!G<6MU2Y|LKqZ#m(eJG=%Y`5wu77#-'
    'qjh}j6nV)SB)E&r8~(Bcqhg4PS%0?-`D_2S~h;5$+Ax`F!sn|8v}fB>$YAn_$-e?cd#zgVP11UoNA_(?y4n?$dL>-'
    '&&OBERTLe6J{Mbj`q{sp6!Gp6shQntU-'
    'd9E1Gw#VW=|G1D2#8DfDwBcM7C9y7w`Cj6YC1g9+Uy+pWi}RY#X!zC|w;%Q3x&$ne0`7N?0G<r-'
    '{MI8beCb`cNJn$gyYd*Qsn#;K$!w>Ske!P3{s>IwL?Lj%aU@_$%T~a$K&e;i8@>bTU=;LVk>^V)%F<SLI^FSgBt5*{cRl5*{=HAq'
    '<5!1_fz}Qs*yLbyNqS3FquX`J2g3*eXav1xk*rsie|Wx839WlEsi(sochNhMJ1mH~={K$wu=oZ*6lw4j8NJ)p-'
    'NP#v_STy^%wWPIn-'
    '1Fdw|^3LEs;Xm|ELlo>CH<w6m{+uCiRWNDg@`4CS*IgGF*2NmKprWJhz@p7`JIW(|L1>s;(!bkef#wf*B)9^LCy?b!ltMw3wZXk*'
    '_0^}ktHB1!Zpmf)IJP0~Sb!y-B-+?GV#vL{rxs(R|*H;#IMm4SykYM6~`DOxT8NZeQ_WN$&>%(9ioZ6hr-VM_~+~+#qSvEW#Z;}^'
    '0<V_}iyVtmlR9+Mz^Z6fX&#qWZp9X)|`e-'
    '}ZpM3i33?fyUgTY}N46@zE7~Blm8%vcrh1_YOz7qv?>X*f`GQ(l!z|(f^&Lbs#_LBT{L&F^s5~E{*=bFYDzS_`5v_Vh;LwYNX{&{'
    '=c3Vm6MgC&jq{M0SG9TAU8as(tLmwH^b2P}_+=mx5ET=KxX$W32lZD2Kvw=^(?-'
    '@lhIyNV`6HjE1m16_PPr*pIn14n$#cBZ7P62l7Vjusymg)EFP!7Ls1+0p=*)T#R2pDP1W!z`mzK;!QN4X81$VcTJ%4>T-'
    'hFPzx=BT&RrFNlSpfV(7lU}o(Cf}o*LaZ)3yWY|Yo#MDMOcoWB2Ml$Q<Iw?EQ9bnMqq`0f|hda-'
    '4SGvbOaHRPt!4Y^n{Xw83*KeYbx@gg=<MUeFjGS;qGoJ#p6IO^Qp5fSd6EoA2KzG{Xsc=k?<ov=*V0psHJa<P=9*MLEs`i6V42ds'
    '6&1h6Ad#^r(UlQZ?C;}J)&w`9MVC6_8*{y(`9N|i!eA@CWiJ0m*@<yJO)ZTmP-nb2DcNMf7<2A@g?Ul#Qb->hdu#w?E8au-'
    '#K()TIjim|$OX%Rbn}de&_LX22z=1emvR<LQn8IsW6awRv1tZfj7-'
    'XE=yhIp?lf8<%7^k<iSkwq|yKYJYZa94`>r7D1vy@%7DC~U#?yF%;9;@jh*dMr3)0>RIBsjde#!0H$o$#Cn4X&F&J-'
    '7E90^AOG6BE5(uU}kGIS)jovocH6xdPQN?*qgZtjtnASFrs$n7iQ}_lS=T(RZSHIkJ{%qf#}CO+oq?<ATRx6`*u2O3`PTux5AC%J'
    'UD6S*zMO=`2=F;@xvu+K&XTiO?IrPiV&Yp+%pY{j;542&5^@TEXrh_3Hi!EEZ4Gri?tyJ6Cicetq?~qfjZIO5%b$Z}Qa?1i{FD_y'
    '(aiCS<_US>G%1r75i3p}`%@IvFW$*uOgX+SMs*0F_DGB`%RWybU)zi}PD4I|{nMNpV7$aTJQO5nhJL2qEzW%2B2ZF<^c+DjYFxcE'
    'N2qVxG+8V-&RFwziFfT0uJU)CxtAN)%0|CK}}-'
    'WR!#e;_ba#0u^~#@2EK^O5reS=}<qZqd>zHejzjdaM}vIeY}uhvdQdd145E}H?uOFa$(lb?XN^M0y~V~Wdm7^u9O)-WMW{-'
    '4Yx`N&fGO_R9F*G5_k2&wMB@q1okB5S}<_vA{;fD2E*^+;K*_B3OZLeOQlwrCWTsEVQ<v{qq4YonYQn1)7l@CKZ6^w+rIAI(`Wb_'
    'WE4ujC|nU7TOnTao)c&IA*cF9czrOok0woBZ@p`l=9TGV-c3@%H>~2QJ@3hY3LuvAJ51UW-IO=N^iziz#qG!d$rF}SIfe$XFTf&z'
    'b!JWF&&+V&JqRXUyFF<`KKXh#2k5t8N{_c^T*@0i5Dvnc&met5%yybbgeAd{dAi>~;A|$zu%2hKiOk`0hzP90K*<5w`WmEO{ZLXv'
    'VEB%yuqieZzMq<LTZjrWve&a7o}ATpU`3NXdeJ5)oXVtOrS$@VyM|P!GM03+4g17~4t_a-8(p|nniudDWe}jS-'
    'hTG(0)%S>$gan;+8c7AV~km--N#%Bd2;d0$>7nCm=Y6t(rip^8UUh<l8Z<b4^tY9L#w%d@2k&$`{t|9)7Aqx0*$q!-'
    'm!?mSzo6Ml}{$KbOsr7@oJ`OdO?5Lwun{gI2QFo<RC_lEZbVcRc-'
    '(y?n7LuK=&*>40Mrww7aAu4qYy8F4Ztx2Nqn4Tgwlhc&}Qmx-Mn8d7y<`j8Qqm+S}*w1tF^FOy-'
    'XeqPp{H?LnxDm~ILHstl{~&I}c|ZrX5nxhrG751?`H2Qa8WJOOiUsQjPu`t^WX7{Wucj+vI-jYZo4jALB%qx*4`9$|<l2CKQq3_v'
    'gpl*90GQtiJzbW1KtqIirohV8ThP&3TQ!oO28JDOA!;4d&q3tc`#tc=8JiTD<`TY4n=oT4?^m!5fo(gWNJB+mi|3R-'
    '+p`Do!;;t%D#MmVy}GUNcgLQ~$cF^p?;swm1K=|+Gk<XtV_(6BJj_+#Kur?7)#dP%#`-7^nbdUD`glzT1269wWLS))-'
    '#6{0CLndd<J@?;%qG;arGlkj9rX`3>$u&@TTGy%MkOr3@+cm}r4I?S%0JC`m3-'
    '=uqN3yOI&q4|CTymR?RN7qnCpB2GW4hMnF>AV7zJ>Z*U#@CL;#_6O0h=b*F;AC{y()J{}@}#TXNCj@nfny2_gsZIHTo3VqMtcrF#'
    'o2J~hk3OJ;6VZ4p#%X-4jXvo`4THwz}jT0YP28Lq1j9}idRrSfGLgmI=ZJ~2%})4FgFh5cNu4@fqd>Om&-'
    'RN$rfbgUeiR^#btBG_A7P75qCEx=dpaeeTuiCifau<+9uQPaevciP!^P`CRKQqnw)zf36Os!usZOTm^wkxf)4zF=mCw;k|0#hxHn'
    'I23MLFEmyIbXG5}rcmGLoHCdAY>qQr&eB797cxo$6C2!549Ry;RX?ye-'
    '<zRgS}=I^I4I6+kBf%D5thU3V<AOhy&*|C~XU?oXF^D&zy%-'
    '`sx_A;$Ngjme^<5r&tKRFpkwwnq?e~Y2S5+l#i?+nfAOrWwdOIVd=pIXU(pr$M>*1|@yn7G>9Zb<gVM7YYb{2m6wGcb9??@UP&75'
    '@UDyKt0Gd8#f0R<R1$e6SRrZ-KF<lA-'
    '94xb^1@#A!?QNAQ`T2aFj|v8i67STBqse7KoVhFHV1D+IX=vajQkqi|!6@Dz_|K<~`AAWiCh;z0#{v{AK4EF0>M@DJ?pWW?Cu61f'
    '2WftT1%Y*lAT2)*w&k4cJxX-'
    '>%Bwz+$eP9Y%R)PpDx8ADtSC8j~<p9nHD^C3Q;LF6fi!H&dYaAc+xSfH5)YM9>KJas>LBWH*NO)6KSo9+_B_P{U$!%*HDM|!;k*='
    'T&5?|}k`<b5zC=b);TBaWxAiWD@CP#2-*_F%O14Nf4yF6ugG(0v)^-F)-'
    'uF~8~ZsVT7nRM#4$2``sG3?*tfwLm8XhbNl}K)X&_T!duMN)a)ryS$F{Sr9Z`5KIFFfkQYk15zVWBbu1J5uS*dpp1`g3YRtzPfX1'
    '*$nSvT6!qW+TZwtYeo`{sc#}x^iH$8q@r(^X<jzv*b%9mhSCLj_DZU@9ww>ua_eNu)Ezgb$@;~5uwNgk1H;4M5%GC~mWbk}3wGtm'
    '~e-xMujNrJdKsqFS4>%2<QJ3D`G0c`^^?|D^wl@w_C&;ORm1-s@i6ybEcR_eiCwM@-'
    '^i)8mRajbyirnZTvP`4AU5y)xTES+Tq8Bmxqf2={Ht+p;7?@jNz-'
    '+IqlGFm=z>vMtnPy{vv7nI2z7A!~un_?2T16lwCF2G_4q4#g5H$I7Dh(K5fnSE99A&>HXIP|SGf<z2JF{aBv5?IP^xQ)=j7}Jx??'
    'p~Fk0XGBx{Hu_Kf^y9!BjyK2Lo*cQg;!c0BLhz@)d`;^W4}0%Lf~mlB`ptrQFD;6t^*(eDbV==(ZN+fCS1~NzQW;h;$IKQkuaOc!'
    'mBrTSZCMtX$v|LT%$&G^bSAZSriN@%g7+I8L}2>neT>!vcL_^Q%%O-VYDNfoi%Bx$&Ex)-'
    'Qp)@sr3^zu;d;V9pdJVWOcIDZrwbu3as*3JrJ`<0Bem;aaTp1=Gyzu8U;&jX7;ASvoIDFJA`)aIo#)Vx?ki=^s&EeNg$7<;($cqr'
    'lxSggr`Xq)^rztO6vi7>nqY5>_beDZ-'
    '0;=<MApypiL@wIkrBWoFV_EuS9!U^_PjiiE<QRKt4=UWJ+sMbgMBbb}chBX_R70ZjK?#bwQl30XvRT}%}x;XsABqX*?wh{!(CzH>'
    'H%b&%K#wyo4bo?n+P!HlnnuL^_B4o`f=Y#PTIA*NAP$)#-B#fvzl#cUj~l|~yV^>dIm>NbUdV&Q3ZQ5&3}OzjIy{1*~-'
    '@h(R+U!eJiqY(bpT<k_F)e7Vj7}a%LvpI^5tt^4)8B0NRaU?4b-'
    '`f`EUnS*a7f|v%$Xq7?as!?hf~C!8z*KaqFoZ=ekZH~SUb#i>&&66lrvZKBX+UmJ%=B9O5KFU+j2M8Ye3;c)Ch{Ce7r1}s50`(%W'
    'n2Js_By7T?v}t~`$eEdkqr(ZZt*bJ2gu!_uAnp7Ix(Yfk*c%&N$XUnOht7)9#V5rSUZ9o%ShQ%yQg~1$OtmY)evP@24u~~FJLa$@'
    'Pz#A7oT%AM<f>xrUHWK1BGg$EWv8eDw*&ESH5YR6@o9{htjIUu_Ta$WR4yO6V_jIDoPPTfl1E0Sj$pa*jZKk&rVSfj1eClvZ%JWb'
    '0e2N#+9`PY3A&ab9XpC`pk9Q%9$Q8V&>K<z*|mskGHUoeB29c-'
    'H<zObG##wq&eK)AYGfAi)*467Z<sszKC4E5y&u1ArHIIMu6Sn6dWMKz17Z8f`)cx+orh}2QK`Dd#F;(P>3enp=|?qS=ak!IG#0{$'
    'Bm3I{=<NdKyPU}HROHg9+f~x<vr-H+9F1$I^8Sx=x~;wcwt6_-'
    '((lPcQD%@8bIDH>h4Njj>vuh(_^~A7VB>81_)wIJe{#qTM%GAd093f#DAQw1S;jr66(fpXOIg{kfmF6;XHa|{}QcU?4kg+RJiw3&'
    '<#2jNE?qy0kaZ$$n<hMPR%TUv+hUc_E6<!Xl;TQ_uS!9I2k@Vbiih>eCp9hH`(Met4&Bf5`m-3h~LIItU1o7NZ0#3-'
    'VyOAtZiu5)j3Pa)^EZxYGyrs^Oq4A>d0gG0p-'
    'Mv8Rg<ZNDQVq=p0NxNcR~m?g(OWBrxv*LyoP8XJZCf4#=BAs=|=(ea;mp=IH0hs#S<6!XZZ~#{xMrR>B_~J1J4Bb~3%TySnPy0`C'
    'E64)JFryv7f-'
    'aHi62T6Lh<4+<5oLL^)5UMWiB3st^_0h;#bI=5r$+&F%~f5M_RvgSAJMjbluWYj6VLVT8QnrpXZD+{OlyHvmh$ekal7Fkqz=jPV@'
    'Ky=N(Z4mNihu{p=_SV<wL;jg3f2QYJQ8Ti={nxHnuHBRRWh8FEv>)up;`q&tAL>3!OX$bWWOR%=s=*!QF?SbZBA_1j?Ccg?DlxM$'
    '*Wx+9X;=Oh4Fd$qM@cxSXcI*!x{*~jK4crW95%BxPB7tTHZz_0v@NGV&~Yz4D~Pyk+yya;%Eka9;-'
    '}Pla5utF*L@kw!vMJ;_8!SFL}e6+ba=ODB|q+b0U_PZ43bFk0CWH?$N_q2aZ$N8LeSiik7a!sao%f?d4JP?j@X7wNj1n@;xxmVEz'
    'd5EP?ii(Z3GmXDlg)e4va8=jl$&4kR0irCmT*h%C?e&2hwh&k>x-4kfZ?VcNj{#35;Ko4Lf~s?%;e+G})-'
    'S_n$q_iO3!PP_ZYR)D!}n3?U_73*t~GDuS0nph8e8dHY9I=HIcxj?C#oo+)a8Mwgr%w~}cXe|ugS{t@K;EJDzO#ITfxVDusvA6PT'
    'WG^XBF?E(n85T4>zOqO?r=*y}ooUAR>0q`ln2DWe{lq6{x{oGJ3X}Ss}r1E5W`!RgOF5UxiVvKp4Tf<9b*w05zGxIrN1=m&$_Vmt'
    'OD4M&&1QLvGj4}YTk#_d39S(ftYWTXyExQ{D`m+gDZ%12vbKR{R(K<Oi&!jB{fwW6O3_~+p)2S;4f^visFJuSLCCxqNQ33|oEvh*'
    '39`O{1(OKSVrmQ<sG8K~+{@(6-be@6<<<t1(yE(Ltmh`3cXq#!&1Sa8eok3wg<%<(hy;|mmGQjQT<^)2I{w(prVLvk-'
    'R~dAr`TC>dN(!S^htY7qc*pzatIMKRapQej-'
    'e0ozjwe<+ZfTE*$sIuufD%F0&A>`8Ps$XzD&;Aq(DnZNqG&>(7ATkFTeUItOhtf^2@hq)g^0)_ltNC)LL^rS!bbtt1**q2T|f75X'
    'dfrT9Y>iJMs1W;FnC<6IKvys=#@oBlv6c=u!@SElja`WF-'
    'Iluq!Z&RGQC_KBBAdggZmwlo#Yth91h?vQyj8Iyer+FeMWF9MR<%LIDv~IPa#Yfp#Q?<q-3^<zZgVmeP=O91?ru-KoTWu!$>&853'
    '{$1GiDr2CCY{P9%H#vwpBWQyWMYkY(SPLlq!PbOm5gFfE9rB4St&a%fNVMFkJ=H%ivJ|F;D&eRMbHtccvKccaa)Wl5ojSc`;IUw#'
    '9GpqAt{U;OUWtU|>>L0vv6N#W?2$GmpV}U=je-lyL7X*3s5YL}4Io?v(82yQ!q(PLozcnQ8kQ)UCpqvTSP~Xi;8QSd>S^{JT@5ax'
    'vM<r%W0CN&y{I`jwBNUavIQh|fde0HDxRt-B*CkPGQGnG1J>Ql^*<Dn3ggDy~T$W*G{!Eh}n2=6>DlU2=;k;@SXUr_J^1iYR*!4-'
    '~yNC^G@aq^GeD#bnN?MI!~4*mrUsO2VZP@x8(beS&mDPGk*1A^Hx~V{>K+p4&)|XvpjeeSDZ_;G{W}l451|dai)POaBR$#?nMf-'
    '7ZFMBhGXSqA-T_7=mYDy1{yfJm9uyB`oA7doBzqpI{ms3Z`MwKx}!_&eYFS9E|p94?BqL&>logcE$?3jVh@`f_axpo2PlYR7I`?+'
    '#p15VVAAdPL?@<H#)$Vkz@=NJVocIM(&~N)SKX|;jbl7inx(&=H{6BjQsF=qUiWRo#Fz_wmfJLHQA?tpZ44u;gbZyIL^YbbK%@>Z'
    'sfTG-0v^oC2%r5^3K#`55XS4b!4`1C1C#t?6$&^NRBp!8n%P}IV1~KV?+WB1WtCqy6vi}Y^#iY9S;Gw_NLNZJ9y#2)E{~h-'
    'OWN=Ux}QiJ(Wzmc`r4Y&UcO`2dnM{HsWrDUP2jLHFF0<kuOURWxc&8JP`qlZV{Es@xhRnpSIx0cC6tkm~D52>2v7;BYO$V3DCpPJ'
    'Vd17W&a@*F4~i;7*{lujcMaYChE|TVi$!Jjm8KlWdvo+89zmK!+3VOa;AFtI#Pubp%n}O{H%A!hXO^_AzT%y^~xk*W|P7AO9NBRb'
    'jvx2c+kGUzAP%L(g^2R(<MACMG?<;MirVnX**B2+d?*bjZ^cqPocME(>C_TpGa`Oz!hoAFKSSRAlIG;ci{7HWCFy1u*U&d3{WD-'
    '@oa2UfH*|26mc+2BFV2`oQYcuQ3evH9Wd5%HE!#RJ=h|Jv~hW4F|f*B0vRim*atJqst`gh?lBO()k2M*5e+?chS!0cR#T@S7rc|w'
    '09!F;t3cU{JBO0;@H@U;uvwGqWV)8K?#672akKXXlKo6aM)OmAd@UAH(yp7929MLb@NMxR9VB(m*z==nB?-'
    'WbKp+8k5=cZlpiScW*Pjn>n9E@~{6W~gJc9?741r_ZfVW&BJNYYtI>_{e>9js!Dr0Z)OrZ3bA|SeO8_=7DfYew=y7hK+<h?3TL1*'
    'WXoc2YqU5Eo>qnqCDnCm&FOcTJQR0DGs;=SrBUrhe*-ld-wisgm2qI|Z{@wtiH1AB+f-!33-'
    'Y8Ij}6*)wAKJHp76kd5LD)6d=>VLH@F{;W&Od`7;jTkH%#KXQsD38DXwc$om6>3K1KUuGdoGOqMac#wJW;E~5U;zzz9R&l}Wu~Q$'
    '<EZWH&F6Cf!qgmiVO>!ANY=4=9ZlJ)Rl0gF&BpP?4j~{fUkEg3HcOgMS``=`FGHB5oTj#td037MlHxj8BT!l+nNyY`PY|Zw^4kXj'
    'x+x10fslUBi`;l_OQunjIpPAjC$Hxg2z8hWCo4gJJ)ZLT$Cr9=UL5c$&@aesdd*sAu0{b}M}=BY!$ZjMt&_F5LCQaS^SjF-'
    '7=ti?Y}21j-|?tN!3?O%IR@c|Jx{#w#%qt~eJc?4r#GpH`3IUbd>L=xT}7CLlPOPgTVq{K-'
    'zn;*UN@1Bd?|Qk6<H@E!$b&9hpFHhpWTJ9YrE$Lpo}M9Nhk&+k2TzZJQxiDpD&YgCKUGYG(&e`^<tK*q@af=s&PR|1KT0QY<{rpG'
    '|)#BqDmI&Hr|=cw9c-1UZ1wC$5|%Zs)X9!Vm$>Gh{YDN_O-0X-'
    'Z=}we?Z16xbBMey%d|i^&0C)uL`hjtpWR;17q99k3%RsYel`GallQM{i2%{nz^8N){DZXkm^94Gh~%Nxyp`^90?fcL2G(bgys|u6'
    '`lRa($GD%x!uliofW{}#(E-Xoaz5THRfxJt{oxD6{3znlT1fnEa%jB#zzsc&)n`s|5K5gE)e{Ki)_ExWhmhTi-Ru!(4g{bNn+lnv'
    'Qox))=fm)l)_>d!{LCZ!W@Ak1aBmnlPSyllP}E6BdSvv{8Xu^!J*D~PN#}LGNlGQjXAi_BI(R+2bQelI5Zi@g`&x3eyNASbPBjFp'
    'elpy%8Hkqz>Tf%bO+AdeWr9^fs8G@p@9>!Bir1`<tK|Oo%ZADL0gp6(M8Tuz>=IJ3d@n`&`FzaPC--D#2!^nIvp0tDIs_%-'
    '(D!alHzR?*7ckPbNuvOy+_)>+7NjPIfV0;j>Fk&X8lHCZ88HBq6-'
    'x0TA)c}1IaSLCZpYaBTGP>oHutd&oVPcWB(d;EU?}(^UD!R{W;*mds3$;svfv~?-RAP3{hJbm|h8cI5|DHx1u@4sf-'
    '}$tVR2X1@o{DQ6d(&Exa9Ny8E6$&06HG(5*@8-'
    'lcgl%uur|?D=&{NBc86!w2O6y#Wf4t2v#rlLm?F@obp*d|~=<;7CGN4&Qu{L2p@hOr#%?WyE4`7>|=$(FxPr9e9sCU8l?67fdeCG'
    '-d^UI!|(FW($>+CONnG&+Lf2ys?ww9+?ChUjpEuTl$=n6-ZEF8g8R&m}U-'
    'iDggVM?dA+2$;41){9T^Q#CKy6Y?|M}yRb*H^nxK4g%{Ih*sZlzJ9@r9vUh3_waTqykc1@LFy1vpxuS|Tut&&bq5|GCvg>t|Mpie'
    'Rny0tE-h+t35mUF$zV;%UN@VJBZQ*cCAMY^kaix0|gDA1U=h8T$FBd~~4ZWA__G4-RJ*s|!jGgx1t!>Gsp+CVQ)B-'
    'j%uXhD`t&Nzs5?MCkRx8Xc1g1@kByG58ab&gu+B6eP#rqcQP4FkSfrUeTUq@C28Eig{$PaSRtsx{Gdmk7`8gT5avgaFAh0mNi*CA'
    '>4=jrwLEZMFl4Z^eTXAB21hXIpiylB>5U$pAQWlnBsL}-Y_he1JU=_JbU&hTB-'
    '6U!?=1VPcxumLsz2Tk+V1VP$dhZ45x$sRS`tRqlr)T9k~+E{aBdoVoz1pb|f>N5@813n6!V!@M~<Y<L)Eg(;t6KE%&zl<vLj5&Tm'
    'd;&NH>SeyWyYzTWh(F9V=Oy?G#l=W9A0s}6ix87=Zu<I|@rOHn8w3wT2}e65#%*RP@i``iL#Kv{`?uN7$$4{<i#c4VB?cYl`7MK{'
    '_WTw#+S4IZVTfzPLAkx~Tp#4j%ou9g51qOn4m7{%j4bWOdoRa6a$4EmIFOSMwU|)!TA%}~@tm9~$;IT15(b{9>yGvJpS}6j<5}an'
    'Q!?%H)|C`5&on+xOe63$*0g4vuKWy>F5b_7xJ5dxVVjchXPZ7Tw1$ZiCj%{0&+}qj73^)^8{PRd;ibQRT0{=XQ#ZwhBQaxoc>EfB'
    'FT8p&`THNg|K`VU-~RC9_dk9A(?5QA_x78g-'
    'v0adKmMnDp{`%|{ZO}4J<sjjR^9key&n7Lf7S9&b^50p^o9PbUjEfA@3*g;dYZ<5YPxBdnrit#b^24)b$!!x?Kn1V*R<!qG_U2IU'
    'cS~eO*@a%<uSjI2iN1!4NX;7-P8>8{P2tVbyf9kGc{w|SMsu&^mBDHPUF~)-'
    'P|>GKM&0^#=5GeaqgSBZO1{M*bdFy>G<k;tcSYm^t=5q$tQ<p=;Nf{9ORvDs)wqQPc`xc`7s&(JoHmvwN=%&%RsuRo2$C*<;Ux}7'
    '43cX?uYOG_Wif-Uj6dbG>>yTOtpyrEb70#{o_C0e*4WobZsV?#xTnGy1whGb{J2yZu%=Jm;a-'
    'qo9m(Prg0eM|J9?uRrh@@V(hBAANp#U{2*eG>9-'
    ';`S@&fW<1mXny1H(Mx|uq8YpN?zhel*k4<h<S;^Qo`(2q8?{{FB?w;Q@vzSoP?tGUsy_A1k17(`Mc^toOZOBQgbdL4wwduTM1sC_'
    'L$6pg9n-MQ_D<;!P*G@A+5-Lv{t%}I%-'
    'wr%RR6=C)|Oh!U&)3<Fq$n0fuvIdKYL_4Z_R_&Tq9TyqN=tX}jd8=(itc$|8q9F1pnYv6&2O(dRS&c)lvrsjbZ%*CVh)NGtE9zaf'
    'D&BSyF^goYd6pk&+eJpLe7BWl5cQSOEHdi5x)%+ZMbTw<EY5NFdhSJc$EqKPs&0n*Mnr^+avRP1NA}q)BWh&1Wjn}lr$uCS)Ab_c'
    'v2AqoE$`LC*o?JoY}r|-?Kn*BBp;f_Nd>p4Ro&I&T=!!?R#Q7KLmuaG?B}r;=?^-'
    '^MPjlX!yr!><ipdTACs}pDnWU2SIx^#F||shILzHFa_#G`>DA`<uXP-=2vY3BTun2U*pNu9mq&D>kE5(hf7zuukqy-'
    '>a+pO(Z70U3J!zR3zOj;RCL5&h7m-bsyfe&V*yg$$S1p^WR$hytnZ-0zI@Cp6vIpf~-'
    'MAVx6~&+yP;^Ly&@VC@b*uNYD6Cl6MJ>8&Y{j-T!%*ph4eM5%MD}B+W={WCzuJgoM1qZed8%X`L}-'
    'KBKUsG%;<9F9oT5To?BcR%Ci%^(bM~U_cbT4uM5iEwJ?YsjJ5Ijeswmf;DF(XhM5D&86AQ7*tdR{Y7ER<TJ8eCW<SU|2V!M0MxJH'
    'flIQ7jUky`(6oEM?Z@-'
    'uR{$otJO_RE1G=Sn^IVhgLT5=*uSO$A<8x;C=e=jEv)QP~%r7>c%0Wm{y{$?LAF)N;)Fh%1@ZHxlTc?b(&gTD1XUn`cp1If09Irb'
    't1Kr$Ll`ZklH`6Fr${Q6bUcW)YYe6p?l%AC<Lj7gIKfa;sv>bmaf~WoPPEZH8%{t9G7OC7Wygp4g*aJcMddGWmuWHq|Cs1sT&Kuf'
    'CRZQfyS$r~{$m60wc7*h@8BvJJ*nUQ;)Sy2?Vzzx(D!UX3ZQ>G{7FQzlE&i`A~h{5D1M5<??<aFWBY)#FRWC5PZRsDWLEy2?xJw@'
    'N{DL*}^{GO^?`U)i`~18Y&BMPA}B&DA)Im(t6jt6!B}EEc~No28-'
    '~R3mylFm!!n3yPbx*nWAK*iKnB*`Bhc7lFxU6cZ@^!=P3}<u!_Oibl_}zEdS$&K#AOSU;Jjn3G8kcUf&mVsaRXI?FL7)>Xc`h^Lh'
    '!OO9CGa(ydCdKK9u8Yh-tR8HpAuE(V)vMjF1P_%53nOLQ^mOU-!W3TpSUSu|i3W%#IkCC&sS!E_WS!{LJ$cfS~wnn@+vFf@5M-'
    '}I?C&dDa=7>b()RdLbZ>p7*Jt@|v?N*Os5SKu|H|mMo%7Jv6-'
    'qxR3q_;e(z6h)qLo3pfX~@W?byI}pC8unw$CGZGdfAYTs+_o%U8BEz^1#HP%Py@{5@NPgG_qZLab{)AVs@HlvB0D3(Q#UAdEY&6N'
    '%@IM&Qj4Iv9>xl-Hvk3iPaKuEKa(*S)#7urilKiol{|nR;wqWACmQ(7X=d?70V|+vG{u3&|gXG#(pi1*;!m?$6vmoE|VN<VsDzEF'
    'E}V0S$WZyc@i@yqF8omqq|!Vx}lNNcUftXQQhg-MZT(4I-FTf4zb&_IMiZw7VVNnlk=yMJvGXNR&Q*Y<cMj-'
    '0u9S@sR@(k%V{fnS(LI}&az&uxVW*RnsN*@iy3IAUQZYqs@S$}ZWe)wI?A8LY13aDhn&FLi@=&2iHYr+MLA_BNyy@An4FBW9A0v='
    '_OkobnyYj~#YIn52V`qZ_Io0-P6RDGO3aG**Na++*k%88gC444YZlj3e5y)9BQa58lI0gx-=f!(P<OX_R!b-(C$lKI>^<4g^P-'
    '&M;D}wQ)X9=jE{CMJueF{(Vz1;BAD4V133^15;^sG%_=VG1Olx>~?X!1h3wA5CN&E_l4b%yctuf!a`Hcuk7F0to5o-'
    '7Bxin(A8jaj$F|4u|hs?Wb&RTtEkW;Nv4`?_Yq8cB{CKGL%*O;ka)T7r!QUZX*s~<IL5G@stNqiS|Ys9Y;rD<f>%Ai+&MLo5)64z'
    'Cn3R!?v-'
    'SnW8jV)gBv>t@LdUGOOSs>BLb<66e*TY0DstRuvrwp^#@NH5}KMQb74u!ruOY$mQ?_SI4E9Z~IV1q7PW2#<Ge2JH*q4^+tOr{}Qd'
    'RB9`h6fT7$$pg+RXn3=v0VLBFSb>zMkfYy*@)s2FULru!dYX-UVf|AeOaq<UG-7oQaLI$?v`V;(ZR@}BI+nMO`Nwu4E+-'
    'Ni{X`nW>QU)(`wboN$rco1~PCt6?7BN;^50ktYMe<Pvd+e!D_m8QU^<d0Pt|^-'
    '6})%7&;C07Qd`tgLE}{5*4U7EU(uPOU^<ua&ofDu`sNtg$Pr3rNrZ+7ppHJJ4@y!8&*_eT*e~O6vY)MPQKbz>wzOu(+xQ+>(MT_N'
    'UsJ^{qK5Fh!s6)#K3ACCWfOL77425k?>xu!yt0)R|)DlBQe&fU#O=kE5XG=)>EQ`;%cj9lC>Yi%(UzQds59=O^$@fDnIql7JO*X7'
    '14-Wbg6lc9rQ3A6&p~OWD%gk1-%}jV#JqSERm>$)~)_Qeq?cylD8HCPK)A;GU@(Y4qiE+M2Ezti6ZsO{_o@vu0-'
    '1;$k&j)8&0M}ET;^5IhLzcg7SN!<`Ue93s9Y1yHtV~+1&_GRcmpZnrlzk&^bfj><w;ycjIv{=c()+@t(!M&;vw8-'
    '>9s_FfZz{0!3o+#V6@Reyv6Ux|b$(K?kv2vPH%D)wdLFYQ+tfP^4ZDkU_m>@!u4(5GgNloE(&5O<LU=63s2%riLl~;wQ=J-'
    '_9pJmF1W<d{NJ;Ra;rhOyxw<m{QbCkG&f)4;wM3rc&&>f;i%T%LZ#lZ*K<#n_vtoT9Ki+F5=pW0!g5<ILETrC8S-'
    '#Sn+E*!|h$>BbHUpg<6f`xQI<+GVvP4vy;PGqRZ7KQ0Pj0CNY>2w^U2eCtFb6KLt`06j+|A_EYwptWu>0WBICjWa=mM>M&I8NxCA'
    'IQMQ|e+LOe>P>ECN$`LrIccM|$JPcPNl+B{CwoF^Zaed<F&ZXoF3s@=oC-'
    'zJPBd+ghYNM*sq%g07T5G5}uUe^}l=oMVO3p3`R74Dwc+0xMWT)wm%c6~XxSWI{Uuneh$eANv>EeRQLev^@%7es-Up9z%;If=5IP'
    'os)lRRas$m*!0D?(mug2+=ICEofd#!9|_*0%(RpDd%=zE&rCQlW@)$&d6%phkS00l%8p;`!-'
    '0E^?A1NqiAAxYl8cLe?6|>0QElc!+9<)sdB`m;JoLPwE_LNF?&96*1HqiJ_H`NTejP6f?P42a!<Q>)9i3OzZh4!DuBuzSw73)OK}'
    'WWm{;dCkOeUj(aCgn0OT8AxR9b&Y3l~RorsW$qZzjRV>wrZEz}Yv24AZH-ns>vQs8-xsjFjR+}yBkp=-'
    'eZH4d^rmA%JuCkWHY0&7T)0l5=S6R!KCvgo__=>JBz`i^~-'
    'd9m{E8wNxb)$C|>ePr&vuJ1~Ll7B?_2@MURmrqsEmTs9w@)h&tfpyL)k-'
    '75)!C3yiZdhUyBr;|w^nHn8W78`s>4u^Ei}cJwC$?`Q)jdCw!rZ8-'
    '9=&4RH*Wb)7h=IL$8n)zg{i!NxqFbQF5|P3IJBaiVMla^>}JjDD9%l;wFd%Z`3sqTeO~Gvv@v@f&z`4ri&Ov6=g{koYFt_i+3%5l'
    'wXpeOiS3h!kA(;7PMqg^y*o@x&<VJlR$7%eECMcjW6G9FrRn<v^J;re^vw`Bhqte81y`o6L(yI0a13DVZR(f@}vD~sU>`Bmdibn_'
    '=2v<w?wfOQP)egafu-2<pQx12}-!OF6S^RT3mJVgStG+b%Mg95}L|xk)vzz-fGcdai#UA)O@rHd{Zrvh`8mli{vL)S<h<vC3@09i'
    '_N-`b>jY}H>RQ$r%TIapNx4~=WU7+hGg7|Er`3s8w$5ndQM18wYqGL8c31lu-'
    'r~AqFhlbJ@Q5kI2Ig0jIsLV^2}ZX<YlwSf2o(I@TT7JEz%v;-;$-!MbHU$8n8;-+38hKdqNU#hoQ2SHx-'
    '8whe83Lsl%e}QpZ>{Mq(m4liRL`N}Sl$yn~T3!Xj+N)g=H{=XzCJb&<ri5J`wLSFZ<Nt@jxc#EArCu@)dyy%4=>Q3PJxF5N-'
    'm2<YKoRT3!lr*msimvL4YSbW=YF~VYgYn7YmgScJwyj~p2p!GIEZ&z#Cy~|Ma8cyV|h=MqQ?J8Y8Ek%bk{;e*zi5c;xp|5yU;<oj'
    'AqtZ85w>)A$#FN+XOM-f_auQ)?C2Y7D2~ur^$$GHVqIsjpZL0L_KEV&|a=?jytpB%~W%;}4`k-'
    '+Ay1x`d(NT!h=5^k^!Z9K;u}h6kYW0xiq#G3DnofYb95*8CPDd+iI4tmcuP~GjK*Bt|u2eHE7E|Qj>Vk>8zRFtf+%?`*=SL!woUE'
    '6>)S<MBS^d&Z+)!Ec8vP}G$BLizB3mL;Sr=Ic@c@Q3uoL6iESFm{iE#x*W$P?A5{j?3dM~}8220eX(Vl4h;&;liCLgLBT`)Dz>%E'
    'Es6Y`tl!}W?-'
    'E{?~npwM#5r<cIX1(W*hYiKB^LcQMG)OzjIX;7k(&MICxAw;YS#cD{JHNt%+ZAD=#(Xn2iHLs9JE^Q^N%C<jYsqVOB8;FA|+gOpF'
    '#iLOBsB0^SxH{--'
    'Qj3UKHm)pB*L17LrUp??u*H6<q?Y%@7Z$rb=~c0YdCyrbddRhk<*LiD7@AR5tWk@uKcPETv7Ac7zE+RZVF61t5|aQ`bU<$sR+pk)'
    '%#UmbwJTG5V=IPhk7BtFtG2c{pPnJQEoN_48(1l1*YMWZaw!tyu^dUV(e)y2)*IS(wG{H}a&C)jENVHe_mr|AgE&2dS_{!u-'
    'SPU%BD_|YR&CrOZP|b7SLm%u)2w*IVv%dT%hb4K4Ozsc5t}9hZBK4duPC0_EIr<qn|saCk<BREQFn~k@{^?X>=b)3_q}M!4J0|-'
    'Tr6DQz@G)3cN%X0kjZVZoD0ruj0-BR>MB-I??aX&P?Sh?Qv4z{6N|8Pi>a{GPaD+>Ry!)^oanNOP!vq|`*K4e%P78@-'
    'U;>9YRr2br|!~0(V^2(DV~^mB8}eEErGH+AmUkTL|!dPO}|j}Rh)EEBdqtYtE9(vIiW{Q4w!Cmw{F^Ad0iy^?6~O_lLVWU3{;l!I'
    '_68F<gBm_olb;pnvh1F?p9OY)J<LU;yMZXdrisG>#Ff2>#^0#1Tnl-'
    'wZf)~ugj4r-nE=jB8qX<X|X*^hK`<3D?U0biAIyc2jZ*^t2-?tRofvB-MGqH{53`H73J(tX$Qls4*M)lmn@D<OtTVydiTv=Uj6db'
    'Cx7|wryu|Ezdrls$N&8P)qg*{`t#ra^xeC^zWw&y+aLe_$9I4E;m3Ece)(UozWLKn-+ljA&7wpZmR>EZ)+4|9-'
    'KW2OqdAbj`ur(v32{wEo_2~DGCd{oncTUW=YRRbr=NZDR>uF;AHI0RtP2D|&Jdhsj@I(bG4}i|hccU{<Zg1~+79591*?C(dHPmf2'
    'aV#(-#kA4-Wtb-fvw5p)u+SSyhD+v@_hHV3)JMuLV2ciUsKPIJj=@L%Vu!6Tb%zH^^Fw4K67lZTh*#=h?Upg>RhICh>=Q-'
    'RqWQDWEFiCsx4Bh36q=B-'
    'SFMy5zQnLGEuo@C{}>wBBePu_9I<o%0Bgpsnp+aR=P6a9=)ZFM~$K`E0SHKSb4TXeCAfe*1X0sq(?kM0g&2uPcy<}nytk*nIUkC6'
    '%4Uv`4l{f#b7wfua6&&Gad*$X^VA{Yp$<TVfXQ1H#vUkI^q<1LT$HqDy-WrE!&Hauy$Rj`88lg#Fil7Mp1(<XbD8|UR^5#N!M$7f'
    '^^^62$%lOF_FZEVdCW6%pDON_QXIBkV^x*os)D3nzBXUm{uy{?Vf9&W6!!JiS{=QXlTFvTM1z+GNr~7k(0%FRMZqvw@_DeV9(q0l'
    'v0YF@;O;gOebZw()A_-9JAP5N|vE$#E9JA7%~o_I3-T&d5y{BTMQs8l>G9=X*&3a;gAFJXNGK$2z^i^F4mZmUpD%j9wB4qXE_!_-'
    'dW2P#2@Rb(Jmp0xZKf^V?&JjzAjw=2)(zj=f^^a`+7e3QRM8Qrefh))*kp#ga|<I$_Mh8iJ*fVQk3r-'
    'eFz@f(f*N%e0Q^GBUXGKx2!{MVX&cepb<Lkr3~1~;2DG<X``h<pf&awXl-'
    '?gxpLjwj&UCik0}KCi|PB~^WsMx;+!Erb(e3Ja(*4;03&XzkL8C;d`a57y9?l!G_5kIwr{*S<oUgCh_3!0?8cdG+qeGj`|$Nr9=='
    '1NKFa%Zp+3?dG!7uHugYyVwh|ETaPJV|N5q{%e-M;-'
    'CwCFv2YqzjB822hZYCkZg51SOsscc){V@YK5V=Y(?gqLdD)jjd`CZQE<E+$it{@3}{p=wgfdAE=Pj|1|+tQ2U7k~p#jz|YO;k?XM'
    'Z-I+@f7de7Cj58@GSF+V$Zy201n-6)QXKimAaYD|)l|`LLHgFtOC~ZEbCWZAX9f~<j6T-'
    'cyc`QqHM$aJ8cJME`lgwnEf6AC(TWtFhkEsd%sspD!R&amj!izhDc?@h=^#>=BWCxXNTz#r@BdiP=m907;Rf3Uo-'
    '~#JmcHY(Jo9CmsyogX-PiC10@vC8mW62os9z@UA5%}*HwkpVe9R<(lb<FaWt##Jx5T@Gmh4-'
    'Hg)gzG<&LYcC;j|GHR=%ges9;k!krHxP4bxPp8Gk~?VX(Jr-BaJhq=^g2~-'
    'iHUM3l@aU!bJ+AE`rJ=|8!;TF!vKlf6Z<>c+Teg_$E7`woN4htGKf(!qm3E{{gU;1^>bT0a56v`bmA*($vO6M(bb-'
    'A`z+!aM`$AjK<+hRE`@iHXDbL4hjCnh;?FSHLGj~yUCy0SnIk4(^T-'
    'of>iF+Va#B+y7Ym^+#SOOMLPeEHu$Hlvj3D0GnLJG^JdcifA_ngyy9+IX)rpk2UZBxERcEFGuTI578mP6j*|aCC{{T2MiJ)2P@en'
    'ukw6<_s{V|C9^~7W_pV9s{s)o%9{|6Tl4M>v|rPLq|I7s(nBb!c(vpwHq=PIGWsKLFq9MavHgwxx9!XX`<p+n0&KA!$6Zrx^E}kM'
    'MfNwMaWV~IG~O3WRQ!H?;(0L+XBoL*$gZcV<6U@NR^S-njNzR6_-'
    'q2sYMDo<WmVsq5nt?+Tlq_=;1k^HD<f{TB=?7KE{*NgqGlwxde=3E4<o0X$*IB(obXh?9J~!Il^N{P#!~jQ-y?-'
    'sbIUl5bmPPT}Aji33>BN)$dMd$?{2|c;(0#!OtZLg7LdUh<qNjDJ71usSs*G2%008!oQcW9YJOVBR(+GUb7!7bDmKq4bq*m_=LHk'
    'e}LHqo8*!5gOKCbN^wf!t<u89RLbY089m-lmtp*V0H6~05<#hKPpbK7-'
    'xMJUD2Ry*H5XA>gR*w^Bas!2MOhrOqcUH^&pj(+%KQ{kJ;pcW00t^|g5>2Ym7*1dFevseqfz|ON1yy`$3}#Ab%r)0kpIY4<Hlv_P'
    'VxfYN$n6FDEx>GkfddE2{NbmI(Fj9c1p+G2QKGcF8C~uLU*t((;;1RK%8o(mF}t~jmVJ<HqXab$yF2`e?AvmeEQkHqCIqEL~3UD+'
    'E+cy%m8kCo!OH`2nyrdBw)l|29a%*<;6R`;KaJZvIk|r`RBJ#7TX4G07_Rk$5=sYamZw^0#w5K;66<ZPSzMwlGcav=s=D=d%sR)i'
    'vmCX=2ABcTWoTdkkT0el5<2ei^PLCo8-'
    '7$Rl`L+QRrl<?1lUoSH<x0K(5Ngh_O<=^0QYBoFqJG1VR`JZ43(15~a>xtm>!^Koic{hw?X*ov>AqhT=AMWKAWNrn>DO-'
    '<K?g)Jo+xrZd!3%*Fx0!A~}tcX?}@`*FZnU9Zj?Fg6}Zr0R_vYIM2-'
    'frI(rWmnjs$40xe_o2*qNh}wN5Z=~q3nfd_e9VV<3d&)GB{`@Nr!lSQBZ!xiHO--cWhw{<ixNK4cQ!^Twwi{o;qBdn(_XEIKy(99'
    'v=JZ|aj9XV5C^5Z*5g6YL8?>i^`9NC9EbvB++nkkOKH%5eVK7*RO2cE2__DhZzfQd@oNcSzwZXVJ`Bdesm;0U-'
    '7x*beXjGJWy9m~CV9a_-elspdyU&j<wX%PpZ}5e?25(oY4CTgkG7Nj$)~^0AX23{7(TYa_Q0uQ3~q+(jit(*LhiIs--'
    '&`c^~+*enc*;V;Ay*d=aG^=dr5|gq2Ue*iP5pZb4}w6Uv20j+8`)_A-'
    '$DG|GYhJg}yAs!IH*)e(ILpj)+GkIRcWBi!m<S1D3}@bOT+REqUPGu1pKCHn5t-'
    'TN;?c&*qD@T(?o^!NNcnAJ6F=EyKVOU$dPlDXYY=Lb{{H$3-CvBTO($M}4+5048;+e)s3ffYdO{C>7B7`#=L~jBD6-'
    'nCJryi`fe&w*CkdvD6D<At>PHk_WhJ{5b+aL!siNMpVhLkFbcTjd1WLj<bwp*2#5JcAz`Jpvy^dSLY9Rp5v}`kA2`s^HG8$@OJux'
    'Ku50ML?Lz2qE*M|wYV8M;fiKH1!gC#5K%nCvGFEmrX_*yw8vB7m>|jdg_pqcgp+yhj-EUcX%AHG2cH-cUxJ#^s8se|eF(oK#_drA'
    'Fa(|j8E?SKkw~&z0XsRul|cEl<yR6h)p6vFJS(Za_tL#_8_@15Xg9`dkdfLekDcp)spDWH!+$h(hE0HKePtU<6$X~j!F4wW4dd-'
    'A!76|QalmA~LU}QT*Rm)C#wiO%reiS3IJbF;Fc2qu6?HLAZ)vfpQLuU4lm^^z`k2+3pqghXyKGU|`vlxq!<al!h{RELa;2s>8G%V'
    '~cyo=DRJA+dxxbF}04V->ZU?-HiQcc*FRrJY2cpt-GE3CC0@X0@1H=}r%u+sAu>CriyWt)8h>s1?ccOYZvX*J1Y&rJbHbfs|T<}<'
    '|0+g;rDf%oE*6dDNdH$g>BOib;H183vX+ILUCPHugKA{=ohZcQq_Rn^HA&{mpYX!T5)T{d^uvk1zn=<k+@0@oZeli9oVaeVKpL{B'
    'b3+}whS5puKBm3bSgxZ*p0ZV6nK;p|GOOs#_JZlQ%d?1P&_OA}Uc6G`cKxNW)iA&@TZ^I4G;`~<1j)E?5Qk)QG9EGB6gqLA5LP&g'
    'pa+K*p449vd3P+5aU2t2Dm?ty&7zM4kt!?9=R*;T7wL%f35=E1#iAH${86_crczf@bKt*2GJ8I5}QaFrSI@C|<D9|v4U&xF<oVEg'
    'QA1@@BY$_1ifRN<g&8!TkT$uH9`zsNRzz*Yg*+3SfD`f@{nHZRI!>tm6Gk1*}71jil#9h5`Z4n|YfjvpN77QG^2uDq(!R$)sLnFt'
    '%E9hL^ER|YeniOhvg}qe+jLPETW!k>4O>2J)AcGsR+rIAI(`Wb_WE4ujC|nU7TOnTao)c&IA*cF9czrOok0woBZ@pnRB?;f2iX-'
    'x5Ib?3=N#;Th5`eIr-(j+z=$^chN!1}jaXT<T0*2L8j-'
    'APnFWij3srVXzbY_P4?jbPg+U?02^10W$Il#UJGkUyD<TBp)aWJe6n|MU?G>-'
    '^Lf+6#Czk#^f%#mRo&vXLTa5+E(7Ga>=fNXgUT9tk%DIqXi$5hx9nF-%c&A2Q?^%&XPSr1Om>U-p9xM-'
    '6TO=YsM(sF^oRYR&%*pfbP(EjRPa-G-UR};9wg<GU~!Cp}U0gCGFXYZ~*xCVgiay+ZMAs0Euh=tmH#FdaH7tfrG9sPhQF_9<D#>}'
    'Px9LlJ;hy?L4qoGM_51Z@vzWV&PZ@&6G?J|G^&{#L>9g7&8^>q{D^2ubD&fsD$Ud>cZFW@iR6|qW;UQ^T#kwX|cu54=!SGfU*xDQ'
    'dK0^PIVFwjKy(QcBCD0I27xm3e&8CY;BZY@82;=M|>>bjKV=7AP&F+}AIYj2yw7euI{Gnqd^gzC<#wFjUoBDyKSr!u6*I}=pgvT4'
    'KJ<!+4mHh{*pAHbjj@dQk@q0)cK5t@Kn7{Wucj+vI-eMQ><jAPvMqw8^$9bt$EgdYA1rN|6GFbtH#@NrV@zddwI?nt6|j5UVsv;x'
    '>OOv%E(Q!+c6QxxDYFh>jBJwvRllx+t0R(d3wnxYk8axvFfpye@3P+~}uDsT{>MIM!p4ZLZDD3+~xWa)5#7NIHc*apU(IaLAWFmf'
    'aK6Oyc!YYwaeH2xSk!YQQSn8wjAQ1?uFmYx$h@#J0$@kBwkMv7>ZK85H2O(r$ao;z7|8cotcsUtk|QreTutRk#=YmCrEGEW+=;2F'
    'j`>oB{1?p(SESd;FtEd=JxgeLb5kj&-'
    '#9bE$(`KCuxIrsxMr}GN%^MG%X>0LV(8|RJ!*bUakfs@f)OWTv^$}_BX;}W<72aX^t$gRq>=!y>1fH65_fv3K?AHvlhfc^wvh7!C'
    'hIc!{Z+Iz2H0Y#Iks?p|GhZf7(m|a1!0Ol{^i{_q+0gHl(!bCWb&1Iaa2F|&!HZI?ZBwKxzdrh-i-'
    '<=$~DazC`<nGWiUh1}XDc*(`t~H2hn_X=w3u}N<)ugJdQj>ExnrX|G;WN8X)B(>+w0uKy|J(8hq6aHNOM);s<CZ+NDVQC+6vsl>2'
    '-t%*J_c)om_kOBrm$RlkIDLVd&xoYs}!>0iMVn%BjH|cX5uh^8-'
    '>9Mq7)CDUta1P$NU8mFdw~+6@UUONdlUW*`#3p{w}qkX}KZ9V$L78nnL)g$T+gylq33E45g76S&e=hXjW$eWt3TVsx;TsO7#N;WN'
    'Bp<Hj2e8)#i3XvNtBem67H5Fc_YJ$s;mnN|LD37XaObqkPKqav89SRjuZOrSN<Uj5U=E#f8MJKW89LTk1NOxVXzQnZ0-'
    'KGQ8WFjYTNJhnx3w#Ow+|E`!<YxKt?Ig(E!0Bf8By^DRh<dY^buLDOth8xqTXx+DAp+c_CAHn>DCfPdg6_7hveSrSz5`^{rgpkPu'
    'H^0#fGUZhj*N}xa(!f+_-46^z-Kup>W>FD$zPZtd4BNlQa)2P6V=Oogfd8CHv#LW}+lUH(vkkB-'
    '96}jm)F^mt4GcXS2t$L)hOOSiUH~Jnba7f|@Lun3*NI9B#3am&`;|Ng^YGn^bOJCsxBJ83ba|XGWVc*R+pC0p@E}xqct2cG6L7MS'
    '$iNsK*hEw`;f^c|hnE<xy<i$lO1+5$rgQ&~vKc7WG(*?n#PY^kT6EiS1A~&Lm$r}NRnD@!}*rrZtgYm@F9HaaWI!+M{Zork8JM8B'
    'm(``11oS)d(QW($J07UXEm0uTF<$VQcRhH2E0X6Boyz@c`4c*}Es3`vfE>A1PWN<5}52`xt5KIQo7ZWM*;r2%X%D|Y8rA+!zU>QD'
    'vF1@N_h==CF>H}9*Y-b!MQ;>5-G)j|`#FCiSyC6KM?K_}ZdMcpKDg=H|NgG`<mg$POt4l*s-'
    '`7kB^dkCxbSck==Dj};1B43<m~D(zl3D=#7qT%r(^(Ag6BN?c*P)CVHUK~^stBOum?Sig4907UhsMn(c))l87Wif8$WcCAa;V*@J'
    '`=ZL$Bbbin-gfThiVv|C_3MGoa_`w-~@FSVeEcJe>gg+f+P+G*$AZWB0K?-<iO-'
    'B4shp*umhG4b}1z}rbtV<kxwaXV|Mi9IR?>fEy@82l(mus=Ohm40Ai)2f+_9_{c*O6lB`*|z2$S%Hr|VDar^s>hd&ufwR>Qka52`'
    '^{azRq=nFetmGbU>co+^;(|y2=-'
    '{G`=3E+*NM6UV;|3ZRtrXUFu4ZTPK7DaUJYOz&lz;hNK(G3gNVx=#b&SiI9B*SmaX<NzCc~N@#Iv{|9?e!Kb6=O^Pi1O-'
    '#%BL)64p12du6`lxQBpO9vgTkFAaTQ3M5dH*Kw(c2-rGZG?^fZB9Iuk$n;G*~AE!q@sLl<QBAaj<C-'
    'l5$;8mj8kRy$tLU)&;E8^z5iIaP-va;sIge)Jr6{d=ia6m#_#e=dbL{Fb+-'
    '#KZ)Ixy@7+g9XwG0CmZ*QJXxr*E~nzQ=*e*fd@<LP?`~lFQ7pixhF@%8hJtQ;bwNJBS)}YeGP-'
    '@Km~}4bDTR_64T%3kkb;mm@ka(9*+E2>%K$cJq{K1v>kUDmt!%9L2g;mM8R#nV|YNl6{A7SPRpvlCrT2_;?;<j^Cf$fair^$?zG_'
    '6rCyzF_8;cTC=}b3Q_wKSwR0pMOi;*`+RuWK5p>KG*J5xOR$Vo7=W96nCU(y!W>8oU>eo#%kbgSthmey8u~e|<jPFIH!r9Sum>@N'
    '8?oT@^{$K@>S8&QgA+6H7BM=@igd%{IR;p0%2Ito1tG|djMOr<$+Y@CWot&}k4d72sJSv=YBqi$Z_ngszxbRhEh0H@Fy#+KA1L&)'
    'tf^ud?KNAbX`35@1>fCaO?5by1d@%+(c@sk`fE-_DdHzE$$1w$SqlF;t7`w*sojAwqIXi$?H9SMFs@oX$R4-'
    'gMTJ*0KG@8)&B|Gtog)s=k&_+ZEvO?O=KANY(dm!`?ociq?plzp&CPu@(NK#E)KPOou4xD)4yJ&HUEm^s>TqTakm24MW+*{JJF~s'
    'd+<O8S?ZS;vDZ(ej{_U`_fvc<QeKQ<S2F=4nM%4acAVr`-G@bGBzEg@ykfMS<UD1h6_evl-oT?{YT>&rSj^yYLrtCv=$2&pYT^Yy'
    '`V-KKfOjp>V&W&B~Kmdv7Bz6h}0_-I(OX-71jMJ4s1$tRl-'
    '1yZCa<~b`bBptv2aD`qqScFC6hL_j_kN18K^FpP<1u+$R+bK#BW?$wnFVmx{g~1u)M^TnNl~}&+#$j>6ROK&J_kwmYMmZ^(2^;*8'
    '8d8@HBM3qA5WNRnj>b4bfVA07i9j<jW8-^1KMSA&f>D=m#_-'
    'jS72fSIZ2|1I)(>O&bye=Lmpei7?}gQ!K{LG=k0*lNO;}DdmLjC&#8>C$kl~4{#1n^-'
    '>IA{F3hpVkyWP<I)p=xGKK|mU#vtvICfH^k=n;pVJ73UE`Y%y{%Zux_%RmFK)T)qL!<{K6e?VWNEX_?Qjf+Ls$2^rG40QFYRA;6a'
    'rA!wghgp&%`Tt#yJvpM=u!lQ_$=St)^49xlzoveZ}0C?!WLj>ex!O_QT3Ud>+l1SH3PFjsGA*rAI+8RNt<brR@6IeZ@aYXbxIRx>'
    'V6OVl97@DQ+%)|i$ge|ia8R&^A!EKnv9%LPc*otj_Hbq2A;k+ySoMVN6c)@wRp~4+STfwMgjr_pd>~V#NYv$HnP9Qhio&J18BB}3'
    'F1|=ldN>xTbR$ta%%q^57V=Zh>pg!bVdo;7(hk*Y*`QNMyTq#F=IIwAn(K8BpH&YjAD^K?-'
    'tzT$DOYYq#Ke!9w<6<FH$U%1NG42qKa#TqPZg<%UUbql-3}3{-%W-u??AOX^^ADX@)afo*x<kEg7)dh&46^T*Mt67-'
    '2?>AGkajlI7g<Zo_Fu*=}&~pxO;KvJB=Pm=vJx4x>povEobeS*H)q9h~op=o%I7{<G&a5qZKND*S{~nnHM!A*AGM9}*N5!cNKiKB'
    '~liOPSCdrOta&x{zy%nwZh$B+qlbM;h1dfnhjEkh`-8MGum@Qc8pIi(G(U%_!4cdRHY2AlX8Qidzj?-'
    'q@ipE17VTwon(prvRhZ!hcYbqh&N{Lv^C*DwGhzlcnp&a0a`G4}^*_>}_rhoX>Csr!>v3=fo9UXEfNCJ9nvQ?hX@4Ft#yD0L+Hk+'
    '52`nMGKU!b&;EPHx%?|6ROOPw&~^?R5_w`5_z5pSqdU)mqr-IX11nNmkh+60VQ6@@SRJkd(6WG4B}f<UgSOEDGsEw+|xVDp<=?p-'
    '`iPiPH=2dDKvg1ZVp}Z2^JF~_=g>n@Mz1Ru%GgUho}}Ub4wYZX>$_;p#y)Gc;T?0Ns24>Hszm?_Nf2Rd(l5%NoLcwi!iV8K3(cBi'
    'F(I#Bpo*Q7j0jk$6*M5Z3aeqd48qH)hN#nh3@m;7uynowLozk-)@azS}MYaOf)DnF2o%k;R|xg6e9Ub5G@KYDo|ai>1w%$Lt8Hy-'
    'Z#pyFe;?1^1$P|yBXd{Mz1Vpp`5A_Y*bXJoOI~mjyWn@C!H8qf9WOXaC+7slY8V?;9QLFu1Oq@MHL$Jo=rt?Aw~FyAn?Gn!EDlwz'
    'fBil^1{WVWOj+aK13mXXOTe#s++k$66I&ZNI1j&l20}$g=vv-'
    'F!d@I;(O%dPVH9d`0WP0>9GMBoKR{Ajx)Jon|M_K!8f>P_AdkDnZfiFOdW$m{l`3uJ5{Z9Tix`KKW2>gyGY$ANv7ndxELuq+j_Tn'
    '9T(~$@bt(+FffTL0gkryVURs6*KjN{4_xw(`UoB~i@J#@3`EJDGTnRwlyux_l4>XuZGTU>RX9_YN$mrz!|Mv`@Q9dycPdmaCVTl5'
    'DZ^hWpn6Kb@-dX?m1Z09X(t>26q>4aRYV1HA-yJZ-;Pkq6th9)W+_C)^|iyCLV-'
    '?WMIFQ3uX}AsZV^RX3jpl2xn5lnK@VbmqSpo`BH%FfH1?sGml<_cq`(sUPR>I~xU3<*?H8d>kaWn2tRW~w-+_9>%q&-'
    'P8)@qdSzDox57P>qtcFqwtn6OT6^wY<KEd)>=30AKKQFh=38Pn!Aryv~O^bCgy_;j9J(C6(#*a@h#SKNyFhd|#d+<D=Ap!ncyKD6'
    'A&HKP_S{Q|+grfAcFqYo!ElDL3%)4~mJk8V9D{>{^&LL6>yNrKe?1uo}=m1|vPBGN^6rH1buZOBpZ-TFezo|eO;YM<qn^NYJ>%$w'
    '1qT>fOeGBl|@}ND`Wd2UPiSQd)uOMjSECW08lvIjyyScgK4sgGBfS0eyw7bi}v1gXD9XAy4o&;u(VSt(}f#ir{s9`(kpTnwP^*AK'
    'JK;UEttlRxKxgHLnR1KF`wpG!P>JL4Y?&PLS8IQH?OQzktmx4>@J4cg$Rrf*`ahE|adyK7!xdVL2*PMq^+};y-'
    'h)_hgK*}ZXV0g(-dvj!4&+x3vwl~7`r}R*ez1-'
    'vk!eMA0V#)AQ{Sf39?WI)=A{xrZG~^=_b!hysi{ghy&jXYng3{xRpCTJwJbPF<Q$2hgsj!LA2?o%7)|J~4!!V^FT!pCh%A{Oolfn'
    '2K0#nX(t2hX6(7wRFlqssL2q#q2^*Sv75KnSO6_GnRI#0NfLN-'
    ';6Q}eXHptohy_VdP{NF=|&6=}+EWl%aG*LerG((|uh0<?dyy8&1XP!Gt_YHag;IQXs<UNB4|NuXbxiOVHOhbGztkyp#rxUC)bFp3'
    'nv#<h^eC@OmuWUNqPAIy|02QMz}6%f6{Lamw+oji3`*L7H@xUy@=_3q>^z@CcPl1~&}4Jpbs;I1zYD`kC5l9lg~)VVu|>JZH4<?2'
    'oApSi{k;~Jy5CqBOJi6}$Y%|V05>0P+Ac#sZKHfQYo(S?x&;6)&ifENj5ogL5t@%-'
    '!0hd0dVupIs%>|UO(gX)98@ovCdu8^H9mO#;C`oeTtpD<6cw|FK{dQ8O+U8N1E%R)dZ*@Ui?_{&^}r!ApRu=$4rS%aJD;g}marX~'
    '|$qEy@QSS9ALrxlp7quhJ*)5@`&&sLDn7BzbF>B}z{5brdLP?(w>qB0+MEftEoymS<JQ&Q62iW?sZ<9d8xupH17_~WmCeS3ktX8X'
    '-WrDsM-KUs*094M6Sa2jlqf|BD?9`+!?tTq9hGSeN$amx1f+4H&lV9pJ^oGz$oBn#BM_M~h%DqWVBrrG#nhcJ+r2n2d8o3+X(tqK'
    'f}mmy3@PIFet1S`h@NiiI(5h!1gOcqN~CJ0GyiRuHv+mwZaK={7rMQ%K|wa_S{9C78_lh;rSgf&d*l7;E56}fq=<4Zj^FYb5sHGf'
    'y;_OxcLGgp;>uAV}Trr{xExYo&9+@Rv0z4_hc5RAbUK(^`6rtf$ZqF@G8NgRV_!{#MkbmMip^S%{`@Y7pUMCk)f8oqcpuq~$Up9n'
    'W_PCQLojg>cjr>Kc~-9(!0A=S1DtCNpo@&l)fRPgu*m26&g3cd9-'
    'xc9Xh5FlYKb?%v>Is!glCSCjp7gjU07*;H1`AG^p8ZgMqt(z1w_rX$9+}RY)1FMFCwHY#H=5}_?^UAaBDW0XMtpeP8;0hmE85UW{'
    'TGO&#dB`}Uzgg<&`YqP?QULeXeXPq`y1{+p_W+D-9zPI4v~now4GjY#x7jbc8K9ZVcxRaqHnnLy2Gj=Xn<23LrBvCWVf-HGCTluR'
    'oF`H;N_5a8Q$x4M<~BJ)V#Z_LDDej2;7lJ6s>5CzxnjimMLZjgleD&3u|ouXy9v{C7ltnI@)u`*|4)T$x&Q(IZm#|6mZ600D~`Yb'
    'RD(*WC24n?%E}$%IW-X}Q;LmYAcsSr3X=kk5WJC9PUa}@&$cjckf^p_@KdFt28TM|2%W0-'
    '$kZ0_B<0{zizF+zomjGT<IwCI7mp@8^QCSE(+A+TVyX<gE2~y=!Z)^{(;YZ-'
    '0h*G11yZ!|h6awvj`VUTd7rGJbef5$2W?6D9y&Pz<8hAkD@TY!=VT&(_H<8E(tsWnMmo(A$tfYe_R79(Vwwj5i-a{hXRjR3d{-'
    'ZlHn28CYC;aVyrtG~c9vP5QCNE16MG-snLtMZO(GjRmO(Tb?dE$>0utoBaf|tpnK2ss*C=0s<&>FU&eR<8p1CJpidX7^+u*?3bXL'
    'nt+uBTlxTA#eo1C57{xCE6T)|Bd+0j{x_OS=%ksPArFmNBZfgF?F_XJAPB5$E%t@km~G!=#!YL-zwzfS3Bf5u|?5FB7NTpyv-0-'
    '2Krnd<QbnD~5QdT`)KLe>x8c99`&Swc*t-'
    'H>I(Vs03ZigHcZfv?CDZMs~1!Bp{QW<B03=jjQ}bfA(tBd4rAGY24V?5Vg1B!RRaR1W>*H3<%+n=qZR(LGBu2YL*EUCg#!hHztIm'
    'N@Sg5R>@6EP_PyJ9rnaNET8sl%VkDd2jv`$`+xUv(wce5RuBQV~|lK+u7YUM01WvFtA6+l%WC=Gm?<BX)}zZw{LyD2knF-'
    'MmLw+w+b_~h`c+lp&O3i<GsN>;&ZPe5M>Vdgc(P4)M7}gq4$#AevdpCU7P2icPBu5>(;hp^UR-MjcEb3nb&oKys}0#T8WUGaFZ0~'
    'h5>V+MUpaHY&bI800){0o#K59_9pmKu=Pa8Rtlk2K?d8qkE0Ulh-`kyAoe~Gk2K)eSu)SJmI|Lab*{J3?9bEd%vrKsOImwp-Om{6'
    'VGaW(!g$fFy}oGGi|d%&0*BBLi4TK{(b7qj-<08-'
    'rze&rfCz%3JzWE}{SJ2KtqB6Kx!xjdS(80#y3IzQbm*pK_9@pt*4Ee_49_Nke<z~)OgHv`k3y$U@Ej&Nj$mBu$1~srdc)_hp2|F9'
    'jzkcj08RyZneXl{JsuMx19Q!J3BE!RE>iu(h)?07!sPaM`udphhkJ7y3=T|KnjzI;`0Y8WghQu>s`s~<%*lCkl8ZT9s3ius<@qgx'
    'bN2ieHQLi5Q(=f}Lnyhu-drE#%*+^S+R2={4Gy%e>5L5Q#(VFyf_?f{i&$Xth|BY_Z3}c@HJ+0*CApZKQF_3$VBE3({<Al~dOT}f'
    'cS@99-nx>{<(bCEiD?AB#+ufQ(*<UDRe-j6Z?p${r81#OUOy-_%d^(glS-'
    'P3KmXwtzO{yFF0Vhk^ofxvCc~okBC+R!BX8(wdoTQQ6B)dG=#Srj^W(Q~fB5nHpT7U;A3wZ%`^`^p|NZ+P|5LtI*RR{U?)!ddhh}'
    'QJd48x~kNxw%YWb%+{nHKlLjP4S|7w=^+t*D!O=B<LorbBYmJd{?KUH1VH%-@$W7Bp`d;UxFTHfj9YfaO%^Eh1|^NZ?r-'
    '`7Jo4)xSbL(}!MzBjdV+swmUjqOx5!=#_B#(9|PVII1<9*1f9k#20Jei-JdX{NfX=TYA4n`!O_`ORsZYx%4GUfYlIt)Xfs8Q?T^%'
    'X8(YhM{iSahm3))1MoMzHPg?Yv!))Y8{G-YnrQO9IILWp8`d%-u>|1-@gC$-K$@|n&wev)6KF9XJP;C?H~W~_S<j%p-'
    'VK$sE4tt<f&a(wZnLtchg_Vx%?j)ozAM4nGU1;zk1ZSWOYTBLs!-P&{xY;MH=-'
    'q)UC{{o$6&2qs+dpy1H&f1R@ZTd0owI)zpLh|5k*gUmWMDUq0H@`un=p?a&RK=uq2>d@B*0ihLT!Nrc&nr0ROE^{aK&w?ozIAVwK'
    'ZqlrW9Ygsq>nutSwU~Y%@S|H74f_3+-fK_u+qp5A1T9&A*dL1Ssp|+7VZDpZbnVhV_A|lb>s-'
    'By%?PpcUMMg6EacC=ftCf9GFDl=5B9&f*Tg$|B5b`ydmFS6ReyxhE$|_nYYM|OR$`e$)t*GZHk`<+qA86Y}My-'
    '6em1XEuJ=;Y_B8OfyMD$np$KnfjuV?*_c9=wV@_YS_hzJ?wwwm=%{kocEM6!Ei`BVX>Rpn(Lijc=vW?A*id-'
    'X6jV=Y^|8~eUmlw+7=J9XnUPAa%Xtz-'
    '|%rt8OktYix>LzW#f_Vd__^aq{dA~9KxVUQ;b^5JRFkI7hPm7qMitL9~=m|7)L9A+_FqL_8p^m_31uT_U<(Fb|#tZ!nG4T;Elc}6'
    '#@kr?X!vQ2X`8{{di%(GD|QC00p&BU~fmFzRwAu<G&nb@vThTDj))!n$N*;KXix~pVXmHdznNaZD)Q2y145Sm43gZxo;kSLKzp<j'
    'eH>R#_>(O5CE!(#r#K8kT^hN03G8`ixziSWlx{%84L{c2MUjfk+(FHe;$gh*|8J?JJ78z6G7<nvLXbz=CcW!Fsdn^os*#$ncZi%4'
    '_|GT4)%$?lS`ccM~Nvnrabp%{#*spm=*YqejZU~|=r3F<^oY86@|bCb22J5|1}Ud+&_n@gS}r-'
    'Q7(s%c^wWiwW@?Dn}Ibt8)D5lOaFTQ3H1)wD{?T%(#itEF3H)^%zN#e#}%bi=gDOwJe?{j8^rtX6X^v-(B?-LpZvl3A-'
    'pK#cRO$Iwt0ZB6-a(Fc*2?3{XDt(n-'
    'GS<k#>N{hf`<$GCgHTUxNv@B^i%NMFyj!Ur&^|Z)KEMTj<S|(kIeN%zS+1}5x^~KZ<V|S95sGL~9Su|!ji^Zakx~t_Ub!RW)8bz5'
    'av79o#dG4oGTB2C;A7T^b1gLJM)tJ(np8spHWR<Q<qe?33TOuv7F@qjv>h8$eF4B@+B%XjAX6;zD)v~ojE#*TpUlEYldwrBB;5ez'
    'DG>hJf2eOzlvHN0p#6ytXezN`JD3k4L`EV_3u!u`0TTikzXWjeD7e(>JY>KHDH5dD^$ZOV3D`IZyN*xmo3&z)-'
    'D4qCKYK_Ei=%WG?PeqnftjRR11JpYrlfy_o9nnM`d$lN&NJK>5$>}=FhNxGGb+Wo557~0E30M76&uggE5fjx=f$7mEbE(7?lVzXl'
    'bwi5i<j=EOR{4!p!Ky(vqx_znF0zAFWMahR%#)RtuXW2|DLZpqzT3-'
    'rJ1r7x_1EP<Ui47TmTplobtWdYWU|v&Z&qiwbte|#Ek>=r$g6%W-'
    '<?JLV#oTvt=pzVUd!af9H`?UuEV+^8#UZAMwLa^tP)eps~Rd!(KIcN)U0Mx%z>D|Vbw1=HRUi6B@{=xIvFrIoWz;#7yonlVyB0S7'
    '$<Q@#XzYmp-!*-lz0-N=d+v%%Oa@@A^In-'
    '?bwQ7o8_n!)s>^7mV=>F>whJ$8~e4`)3d<NzQ3rY7|LGF?4T=Aa8Qc1QqODDQ$WVQyd@h{)<7pY$>F|itU)|IwJCC@>W*IKE&3)}'
    'BsO2vMU1G<TMr5mmq<=ep?N(hW&71?B`ej)<*SwY53)i!pd}8-'
    'xEEp#MTFwHh}~RF!XmT29yM&3)$^4H<z&`gWY*jWP5cs34%zLq?5V-'
    'qm5EX=KE3Q}@dm{qRg1Cu`Lf~?Fv+iqFFe+()Z}ZuI5CZmcwF?n6IE@*SJL3B6=AB-'
    '#45@nHfq`Hwpt@DSst+)t?Yi$;&$C8vKZnmh$|uXO&k`Tw4TWlaERY0W^P_LrWin7I$1C=vhBQBvQY$E%YUk$Ends>q}M3(+HvpB'
    'X6#mK5_rfNt0tyaAN1C}Xhce4DK!2PM^a;})7#=@%4j4k5$&qh2-'
    'bM_7kXTbx;?~dsiB_Dw`95!gVwUydfBAvX{ZFm&FXZC#5)j2V2PjQBeK0^&#N0Z3~S^hF-'
    '9$FFH)6QZ1M9%DJ8^`brnBO_G7c045IJioh{*IyB-aqe!8(+4MRi%>f%nXbzjK&B<4caXdZ9GdA$w$?kvfxsJ#=-'
    '7GF^!CmH8(THfTJocIz?PlY8vB+8@Fw0Q0PqNQRPYISzSU6SZ%4RIv!mBV-'
    '0R1zsK>%7@`>Klsss9UNYizxl_MNv>a9mNglR1cQP%<@h3adeqvX_li^9DkXYm?TkLv1$6%PL2{eT}8?Dg-BH-IKNg^7G;q=B__G'
    'Mkzh4lJF0^vL4bQWHgA=os8rW!yr<D#+m9z1ikQWi%Yi6CKxe<FhEH~p_*t?GRaxp~kgYnAlTJz2s#}F9ra|H{*?8><M;Y{psKs2'
    '$k<-'
    '@wA|27TzE{^!v~WK8zv>am^u$QX3EwOSmbmC@0cVAY#HtL73~TY3WCoQQqqfV6a9jj8B}%9#lxT?f4sw3=ZA^lyK(ibNjS6sFkGO'
    'd;IpRv`G^dkPMQo~*bFDGN>dcGz9#roon9^8dk>IE<n+T=XKz%t(L{((liTf*hEhe%)c{-w+<LZfs&0K=eTBWBuS$?-'
    '$9a~Xy8Js*o#wMQVYCXga6+<Z}rkp)uh8Gd4*%ilA4wG7)=d27b(z_9%%0{=8g!fl2u^}`=<?KCfes|+~F9xlv6dltLK;4N`u%M8'
    'DT79*_tM!s^s4ygKl1Nvcv7XZ6IgNU*$RRfCgcMDXb6DTeLw}W5ucwCWdkF~ZQ+(8m6%>C(U6x7R`{jg}2h19~Nn9a=99Mnq)QKN'
    'Odq;V&3}}(I%v=_zRewd|<oZV1!$#Vv(G*Len2vbyvl=l^M+5Rr5C@g;q<*&8I@xv8N$R59Dsnkr<dfBW3LW}Rq$xhIME=!+R!r@'
    'rp|l)alSpsbnQ}@=z$`AYp5O8xC&*I`Z>16K>7i<JCs~75&81k^W&XNPHAWP9cFzv1+Kf@Xh(^z)aaCp+u|$0ma14vWKxIzF+o*>'
    'jR$q70JPcPd)ZnAh@ynKN>UnUFsN@TaJZFVbTGhtYky3*p2f$FP11%1AcZ$U1H$?`WI^wH;q>;Z!NIaT)SzB>-YsEWN|0Qr-'
    'g*nQY6^YS^L>!>i<!%)=5Xq@#FH@2=7FDgpYf#vyT|cLgwW6$|6{DPedbd%lvP(!MW=kd}9{+rE#&0ddl$=iz9Eh@uNJO_}-'
    'FhTWBV5iPVU1Wtg<cenuVt3vh*`0=4pY=>61Pxy+v<(Zl{$_h4{=-'
    'PlV7hfk&3XFLqH;;<s6pf6d|eoRv@Ao7E|17V67g5{O0P{%keJDQZ1H7j=KKj@QTrH7o1H_xCIo`Vam&H5%;({6V1Hb2Fz-'
    'S<Ro3?Euq`$b4*Jlg-vlPaCLvg!x%*OHG&1V9a(X2mD!Yb8W89pYdx&RT(sh9tt+5tysmVm2VJ$A3h|%BNVMV(slXSLH>p=SDGs4'
    '-Q#&o*g#1<Zx4t2Yzp8hmXO}LPXudpt#ZzR%%I9P|iH=SBYwE746Dwz*jB<_CM?K#=y?ap%rD;}a>p7(Jm93`tL(mjk(iZutS1C4'
    '3&*AP$+=<d=<!xc(>AR{|vbefO6*!XVPb<zZi=rwkk*b95tN9c|Dt1F+CNU?qV&xjLi^ZKqxx{A^wOb~sk%BtU^C(}OPEMVol)Wx'
    '<tt<KPi%z`=g`jGM*_$P3=+tEwb0kwA)CphVs#b&LS}z;wewD78c6IGlz5C(TyKQ{wW@FUE6`*xFy$-'
    'Y@3Ei8rd3znb==ii=#>oF@7^<%A&<v{=tf7UTP>N=#_qBKtVs90})cdM>ab-unycospAH_GUR}fesi+OPamW#LrhLDHp-BhRG$m!'
    'xx<DzD+6@*dnrn3gJ3S0g<id!fvAWq+KBkRNqPA^c!Z0H@MY+!j>H;;K)Uxm90UN?GS)wlCpTj5B?sgUDIq%NP82yl%r<=LHh`{N'
    'Rab!+(FC>ALmz${i&LXahjl$cbbGTCi%_Rnje)-'
    'Jy`uOD02MV=%A(i_NOmjCI@M=ThtoNcm%`g>iwJatx8n3vN^l)F9|ib>I+T6X$4UTyS@c)Qd?7LzC|v-'
    'm+hRAdwWvpo#k3aeYNoNZ$3X4yO^d{eO(@vwS5fL1F}%fa4j*eRP|Oz0wxS+PKI1jITo_D^rt^wy_Ykc8ERk!OknET@5Zy4?~VN&'
    'wW#X(l?U7{#z&UdTDC;ZHNlPFRnENn*ZPj=@U7`(Y8RY}`igihH#t`T`3!4S*GqlZdC*ShTsi?-'
    'A=E(Y@Y%O8hOZx@cKewuXO^fYw&9tg@DQk9w<BtGji&+-'
    'b#ml9|epQuT{NAfgbTRE}d&AiWn^Hn!LT4HPsg)?;Z2Kg3&DKvuDL^7uvAt*(WRc9>W9SVq!mNUTs(B{8A-'
    'V;XXb<2I^YQjL`rT*Y1MNu`&3;uOu*m=pJMqf&>`EHwDh3qNs#+8X^OeaDWYhhDaB-'
    '6+^D7G}i>X0`3TnuM;C)mk>J1TxEp7CDIoSM*E#K20W>2fZulruBYEZva|dQxV$g40bAB-IaQ^J*>`Prv^|18F6RyjA$3_lYJ)!y'
    'VYy(YQ5oVmiVpGC{x_+ZWXV(bfOQnI-'
    'R2VIq~+HG?q*zf|bUS)#lXCORflPWtLqcex5=Xa+06W0tq83naI$z68sGdP%B};tXP+rG705|<$hPJsct~=BxJqBU+$OiWLz%^<P'
    'W0u^LkyPIyxv=E2}-P;F2aFs9&ZL;MlFM$)ve4lgy*hBW)4)r2A0sm6w5^)VbA*h`!Y@TB7nB>oHt=7z-'
    'FxZS8VCcSBTNOqi&Truv8n(D0Voaxu1#S<ZjGO7B&udM&ql9SS9$ZqwB`FF7%Kl`H~R*GyD(y)zPd$r-'
    'KLE^=<KVCAUJvz&H%LAV-y(PLG|B@1EJZNG?7{arB(V(@hJ7S~Wrk~rJ_a)&WnA6jKCvy-'
    'i<mo1Y_<|YyxZmt}zujkK_&Uz7Bea&P(q#rb#7Tb`BmnDa8qsCgyW_|MTmc$P+2x2+qCs(Xrx3maXR9+^3GA^<}Vpdc)7Z6@&TQv'
    '&v>s7*}D9jo{iqWt2+GhbHn#>~<I8nnOhD{E<u2};Z@%t3{)Z91u(UY_#1XR#kf<dw3Hx5TPZ7;<x;(m7L#!2@>E1Pw>jJ%R}63k'
    '@<o@%xsaLba2?bVA06^Hl%-Ms3yqTCXc=-o-'
    'bs>XsBiKyjhoY%XqYJof^^+05y#wD7QP`Xz27aKIRr)$JU2QN;w!pe2E8kczi^TlbO+9kJZ5x3er`EAVt7{^82ox*#QX25B(+0<('
    'S<4^Ct`OB+czWVI%fBNp-U;o#)@BY^pfB)mVzx?pyyI23M*^Phw`!|1n`@^5V`G+Q7{@1H-{`Av#-'
    '~Uz9E>TLRSKq2N%5Q%6>2Kd?zT~eye@gp6T$7Kd?IDItBZ+*zcdiclU;gmvXP>;4@qhJ)FCH-'
    'y13{WI1ZSBewmh?pJ%7ug$fhZ{o7}jz130D4>R)f3zLnQOqxkYSkB`5%#<AgGYcjd^>998MQ1q!hzy0k3H95Lao*CWO)bk_HvNC('
    'G8Jz7F$A3nBBZaZgoZ9PFwJIKBCAYUamuVMbq+MebyR|1-'
    'MPG$#i_~hp<R|iHxft4##&edCiE6GI&|IVyH($H7=Ei=cD^FRzjVGphf5TeIYJ|7PHZ<|5@!MrZvTGD8*>=d#SX;uJdJGW~&sqSa'
    'w%yZ=@L-2EcFGOWS=eF<LrhveEl*-'
    'G;zvP8^d|uVPdc$(<eF>dRM>Jn*h7w=F4JoC3H93EsjzN~v}^~yk`4f{<27JP#8yAxMzM<&Z2=VF!fFD9y{0Ef_m+*Y>F*p9Np={'
    'fPR`BT5y4?k4D|fCbgkPtNr$v4TNsY1rXt?%xgI+9tXrCBf75_2_S?UekhmgKYCN&P$tsy5>>BE64x&@IOev+^DW8+|#B@?-'
    '6J2jIz%h%>rDPe3MvUnFjUnR@ic{jW0N9vJzR3WxLdh>*oTh_+7!Elge`d%AvCs!K;$n>{`DLTI=@Bw!ewJf4<ejw~ujR3>8toF2'
    'h|3)vITnJM@9WY9fY5vUdVVZ)xUc7fA4Sd{YAY6=WleXJk01ijyYhiNW+LbyhqUGUMjwKQcC>qBBH!K2+K4Hi$1UrSTNvyj9q4`z'
    'dnW_NGI$0dNZM#~5a@?J2AW=7Vy;}jwqx8!!($3T{$l#R_`LX0hd5`*Pu=C4r95B<IlzeT>SOug5?_+`?vC~1Zd^3SoinkE2Nyqi'
    'afq(|AMD1NZQ30m{Y8Ael(X+psE_jg!m66_2aN-W>$P&*p{)djJKQ@&_z`iZ&>sXP-pO5r_dy?>w+JD*lAB40upoCalBxg@Yk$nZ'
    '4MeWei@SlYhzfnaLw=X@`8X?eoGVDeUO#(?2jG8orv32yg9LZ^B!|F(Cr6|M-'
    'Em&#s<%L>_jfHLjlz$2AOrmei~L61Uhr=CA;p`23?j!gTul}27Nl?OyksI%F*iA*pJpII$LM68&C9U>Rijg3rpd(Bq;DDw+5#bR6'
    '|G3&d8k)U$lS9VAIy%oAc-'
    ';GV$<m=QkWxV_oPUsdv)*sSkU|dC8FUL+XbF9mH(E$<FrBZWtyrx&KKR+1O@`v*`Ak$X#%K+Chs3pPuMpJbiaJeB!H8jCLm=S1yH'
    'o)JA#(%TZe@&v8iSAWTT$+^AFXiYvlXAUH1xiK7=&MW2$@Z=Tx_Ma;l#Sx@sThQl}+QMTmNtWV~jIs8VaMj4t+YTQ!H<Iv@YsOJ$'
    'amx99pDWV~VQ0t-4UXyOPi{EH@pBZqwH*Fn>{=$}z2chH2amc1yQx4_lqT48Zl6nt&ab>YZOu^g9p84}_-ayzdRlN`7g+J}zE4v-'
    '&RS)hkUCTKYC;QGp#ADJT(Xf7Sh9nFEIM`dKb{O=!|QOa~5I>_@K-m~L7?nPqF0#yoayjK~}E?_beGL$-'
    '&j#FzKn0q}Z1D*>wx<qj;s35*+UThT2!>1o}1{l+SN`?dr{vr;K0a&?C`i}bvU<UAYJrByEBOP|yJ|GF<DcGOd4H*j@O>VNF^q2='
    'Zja<)MUPO^JQSmEGzS*E*pm8MK%M<P*BM!+TWT_+^(8hQ&$VJHa6+N150p^Nq1{R7j5bI8)$|&faKOlD}P%?R?7AfSAPbDaY{v$P'
    'LhbJYWhv$6OnC<3ksdnZ27*9?UT7pmJ5-'
    '^Uf@M`y@G2G2bKaJ_LH^2Mj2#*~>c}4lbUbWiw<!_f`=}K3&Iu?H?A#Yv+{qBU8ET0sLSB{Jk{9KYC7{5D&$mc<uQYu-unsr(bg6'
    '7Dj@b4vTN03>;h!4!P*X+m2oM)6tgLJ1XK4EU?A7FODCV8a%AmsS9Qk;@_tF&-2mGb#$MvwQ?Wtc4l094{$A}AH!H=WCTQ-ma-AS'
    'N!<2wju4vmc49=ps>;<nX|-<E)G+^HWIm>N09J17M(XC-(4i5#zFg5C+BGWi*N(`skCN?I7obcXfs~Bar{dRpZ8G=}z(j-'
    'bwMilXal}H#R_$mW>~1vh33fj6Jg0#sDAOx~&%sKFg!f9jwcAnAaQ-r<!S{yJ|@zawLPz^YK-36-'
    'CFN&&3v>e)g{?MZ9}NYG(J^S3S(k0B(Do*^@;L3gg=(V8mVqk!_Xb#XG*>#Ja+=2W7zd=eJN6+Xig_N>@ix6oQLGCVLg264nRzX<'
    '~4)#*mV<K9olXa_rgrbt+pF`0+QFx>?v_le>hJ&Ipi*BidUe{)%{$9G9zVxTq%zolKRzkRRi!7(O1zRk;{3R;pKi_NsxCga?g42t'
    '%QbK|xxg)cK259n}G7!a4g;{${chwhGcvfs!L@DycNpZTI-TWHF>xDz`D6p{8Os4gd~*veCTDTie`^1IFrlb>4un@kk<7Z{$#;(;'
    'WyL%m**K!UjDy+MT@*WyVWlxln}gwsu=6S(@f!KEzW{4kIkdL4`PtX+<AFyqv6Q4h<|*K{!~H@R7c=F-'
    'o!3G<*$j?;f1?YCQy^8;GKf0J(@u4HJbpDBZOl4}uO-o!WQ(cOVLoafi)DE~P>L^_9h)QH`qvB$zm0zL`K-'
    '#;+xS{k|Lc`Y;#=r#9!Zcf<4#_qon@mJN@`o8$!#d6S9X?lo>Bl@~?GeEvt;vnv+Ur@`N~KH5(9C!hX0gGiO;VEEXE(Xp=#z^Of0'
    '%2=w*DdbKI^_?iFQ@<>hl^G5*2cEWTcOEI}vzO$r8yfDAkQf~cJl8bN@YRMcq78x)7}8s5^v~PlR_M!894u+<=cjJj?TC0(k|Q7~'
    'xzyvbJz#kpL^n{S<B|v7WqOA0UK?1=;w=qK;YTi)c`JvzjY1C=2D<ooPUmPD29Efe?Mz8oC59E!9W6dC3RxIof>}E1v!wwrsZ;g4'
    'KUW5%hFM0bfX3ek8c<_g!?wdjA81(2UO2J!N1%wMUJwgG0XNqhj=RR6BM>wcDo$!dl??j`i<sI72XEpy%SdLOTqk7*x&sWloD_F;'
    '{&43x?n?LA2aYr!B{%|ar#}dE<oZn%QWq^+b$ninn~@W)Xy#L3cESn~#WNfmZ(?Rz66j8QJQa=!lAK?72`o=Indk24$s>{WK-GTm'
    'i6QYNs2Pn)W$)F8@JnLc9z_5{;8~FI2CN*3B)b)`lOtRSluuiJB@t5{N8ZS@lG=MO-'
    '5a+7?XH4$W4s0#slD>pxek~*4mL9UM`LH$1gO?mwy{)UU<n;ucXQA%-'
    'o6s70yq!{Ox7!u7gKmGi$Y+WvS4I727`=qo0kX!ak5uY7vuDn7K<7Mo7YWgzzwI5S)B>0d6u%v7KOb}z<o80$zwHL1p5P5YI>6qm'
    ';{G6*EmU4yAz)K>sSw<a}>|*fHyJG`}O+8^_25KRJu-Pi8@!H8s>d~*n*W=%I6BUUk7tHyyG76u_5|SR4+%?GHsMCSNz<D=wplv9'
    '*b3g(zPf>pJl?D-AOCYKQw0K0}zJhJ;F8ZM*`PG=#AeeG-LeGqR-'
    '9#+0HKn(iCQ`V0Vyub^inwi>GN*Mjqy!^X|h>#=s;o*<0b0PbG1|oj3Vv3W8u{KYW8w8xt~M>8uY(d^u!k5)6W8O@W*bL~+CZ)xp'
    '=UPFVw}OxiASiQM6BxZzox-%8n0&;?G46T*z6P?U}EGE7DYi7!x&GF^xP^RrRmh;g$EZp#t#WF{Y@pcS{ZZ5-'
    '4J(vhcDD1ua?Xfid?C=Vf{Bm@v|@7)rp$jf?1%{fsDhfzz1`bixH8m90Inem6yR^aX9g#?pL1wtDTlH9wQmEn{Nvwm)WC8817Vf-'
    '!|$YOM*%m5-215<9eRYGv)u5qKnnt+nHs~4^<LWCu-Cn?v0fkPMJsL3>#UFm#i<hXYQovWLrQY%c8LanZ_w`zb<SzNqK+xN9;?T-'
    'Oua3glx*S&lC41a@+Lg^QUD}rMy#B1Jj;tW6JRKEzX561S<q^awzH|(Y);oDPjM4tELI|YEs`5h+7iEhCg;q|FQ{NZ*SfaD2_ryR'
    '4A;aa#Ee^c=(|LM#OXWau_(zV-@B;-'
    '@AcXNPj3nuV*d&VWW@xxwN3O4Zo=4l=gP6R{d>3#!YvY8UY`kCnjtl@I_2due3sQ}q(8ni0?P*O2qc#5g8Dee-'
    '!dzx`Kh{`Xrm$4ptoYi+=g_1pb(IzK&$|PN-)d7K<g;b}oC4JtY*L>*fyavB`z>Ou`n#>E-it+_e5N|(wce%ke-eXtaS<MW&KrzN'
    '6)b8V{gsixD=47bo$47~YJZUy2G7Z2^Mu9~nW`_w3O<HQ$T)+3#=f8dP)#quW0UU3}nosXo#Ne#2dlr{ZCbM(~^>XoQrfPb@df6t'
    '3Rbup-qV9(rpvX~UTWh$=4IsjOh#3{=o`rsap0SU1PjtkX%SFkh8iuRBf=h90`Qa1qRhd=SrJObov~Y_NDQ8%FyBfYAAQhd-'
    '{1F0DcV4YMcvKOuO#vL0;V|BrkK$HI8}2T5LCkmkGj90+1{H`WU}g=K?Ng4>1k}P19+Gv;wCrvp+6G`8<60lxhNEN%Lp&h#@K-'
    '2BW&na=pd5yelWPC%p<8k#62)VzF>I$5fSO?j7XF=*+0hiD0DplgS?H=6Vr8Y|GPsq}BhkYYtw5ZLDAHqyWK7@yKZ_YE9|^1hOuL'
    '5#2S<%rt{b2;Xv#a3-nbB_%9|XzZ3K2g2Gw%SfhB&%9|H$Ag$o?h721X6o*B*3Qvqj{+-'
    'o78D1g?;|BP~@5bd4Gj0M_HCo4*$88|2}gr`+X8;+SJf;E3%gs?<12^y~88MHd<FuQ*4T)GG}lkTxC;N{JPX6_9z$mLrWUBeiCRs'
    '>Ty)B`rB^9qpffNzqyTRRpTCx!x$4HmwElhIvE+mq;yO`oa=jLR1%?z@2liR`Gh?Y9aB8f`256eqm7AJEkv00ITjh7#B-'
    'IfYzK+Iz2H0Wp)Qs?nBLho&&u&|N`^0A?-XOXZ%5af^b9!t^(g)MXsD2Hd%?A}-'
    '&5BwKTpdrfm$@1=yk3Qk#)=%y%z%UB7Ap7Bz@wM!LmCTh{s8k4u~mZv>AZB%tos=ieHRce;*MnG-'
    '(L3|Pz3R&QpiVlu}KO#N65!x06&KcL|sg=Pj;UreXR1q1tuJy|J7_1Fq3K~)V!g3!zCKvnNUXl>}Dn++=LayAENw{a5nNiGNQ(+u'
    '~D9HmSnwJyDF@Qm^%!jpOg`mJDlGx{CHd~m#*h}qYT6PFwoAbx5<`904G7f?_Wr_Y4L-'
    '{2}x})C?n$?*=Ic1imDosMQasWXgSz4imjbbrdwYlAp?2U<V<z)Fi42EaG_lVS)k|e751(<l@D4+8DTm}MTRjv8BDm>o;V@)MPaU'
    'yZ+&l!l*mby+dF7CFB9*`qbYO0s0&<o=VA8taFA=dEh3SlpU;p?~@Dcp@CJjElr&O7rh$d-'
    'Da2vI@vY*aK7ONzQ9AOzbw84)$OZ)O0Yq}@&A0{n@s?JS9@_x&cUqj@M#pMnWd$R)QqeUaL^=pBlcA$Er{)*t~;2{N-'
    'Ik;u~pgA$3w;K+n4uvG$T#5CmQ`TEJrJ3}OBQn^aqbPE`^2Zk9KhVs@p(%>aXLF3ze4-'
    '_~gMTB8C2gRivTReqTq!4k0uLyOt2cxC$bOHf(QL8zF<IAw!=9^EC`AwJ4Vu{tEy4E1gjkyG3D5t|I2|6J-Ji|-'
    '?+I2GMBFutTHi^O4<+Y*Df}rVwVD2Xf9KwkikQ$Li(ZuA9@I*}iWPEH>zqEmPVrq^-eg_<<_y#xFN(`9sbCc<An?!a{Y-'
    '}ltXKVl>&6di<3#@Xy&wAQ=GkGVQLX&i?yEhsW?Q?cikpBUft(8JDxFghEtJcyG@x6dFSOS<~i4V6w3X}%Mblm0OhlA7bId<vY9Y'
    'beH7Am-!VSD2+b%LB4;#ZoSl$FG`-UYrv?cV|a(o+GaRw1f`YTf8svP^fpUELar`oCs6p%>Bkqf2={Ht+p;7zkTnz-'
    ')u8lGFm=z>p2nnGR!sxS)`}z7A!~un_=iRYf2r$E2KbEHPelJTz`T!2<>iuof^wOOCSNk~8d1g`2n?JEjr~*_=S5Jyh=Kgwgp1<Y'
    'dP<0w}1v2y^!{{KN4~6(n&m&_*D27Xb>8JqIRVafmxlh#jzeuv;lfK1Eu}jeJUR8?&<~Pd<olYf%nJpaPa8Iwyfh2N5e}8%%*$=#'
    'R4nmt+;o1#cnLHl9V3B(JaU^ckOj+J)nU%d_@?0DWP{t5QPV4-'
    'dqFYPt`(@jIQ??+v+e`^w++;)K7DK$s~e!bC$aQj$dxUb|Xs6&mnl#z%C;!qr*n3#NnFT^Gsl8*|!LvUFaQUcL?p;9z^c#frz+Vn'
    'Cu2`=DYe%b5clM}fOvF5BZeDYh>vokGcUuw0P1c`PDdN;spi0}1c*p$~Yw4~)F7lHr?)^j4dvM?coi4V9v+DB|mK3ejxDkp@tqx6'
    '2S45nCX|pL?#1vgZ9KWcbjnFja$u;}PO29#l;s+WJJx&PfZ_d0{WuwjhVR6h66pMh)zossA;*H}Pe#X}D%Yl1A|)mzibPDB{eO8`'
    '<Qh7%6ae05$5;gg{&&e0EVAoOMj?3rzDD5_a(}N5o&C)rX@H{taC0;wjY%bjBM+bX+4jil414Pv{vkLFI2GPY>U>7N%JxRbv<I@j'
    'S>J??1T#&kMnl<TIElI#n17BNwl<W`8ek>-OiGte>-eK7azOJ-Iz@?922~`w%OzjKmiJo4B6*bd;7Jk@RB%&w*?KrZ-vp^`V(ST='
    'o^0f#I^(^e<3}W#;aiSK9_mMAXgktREnYhq{c;<n6@H1&d{!<yN|(2^|lqwJ59^L7rx$+$lgzLuCS)G;4^wD+8`(0~&8Lo!*h3{o'
    '-@3)`;Z5!4yCcj-b%YvhIszjM;2WrfozBx_l6$HyLJvraByM0_jNRd~m#B{WYhw6g3o><h+ZmEJcEyb+-'
    'TPl=r|e(!1HAJ2!H9V_ZG^(d|C2Js%Hd<{E6}Oby&m{)huq=45Ml3;f6jzQDE(8FNsxP##K-!~G4?wYj;VCR%TCp*w1W$h8-'
    'Nq{9^SunTSk;2lo20W#cM)eI$QXlJ%dntOBL;$OIBD#ZYWn8O`<HoygRy>Eu{l%@T^krBsxHyrN^O#;oO>7<bNopn@#CKdGficW0'
    '0SB}x)Y(DV<j5z4Li{3k!+Yikk!?nA~hP(@TIimIfypHKQTa3D~iy(*~@#Mu$B|(7a<R$u|LH=!UNrwvevh=y}{em2Ef-'
    'K#l3+E9c`<H0-%6qunVgb%nXcJPv4Z0jiON-1xT|_L-rk&eSYi6k&4e?j2+Jifw4p@+tin{8LUlQ!g*>T(6nM>r=RXzIfCsW)r5&'
    '@;lh~LIItU1mntj)Qi5o+>DUmVP7XKh2f-p*M`wyqPFS9=u*_HOh|WJa{8&5z+jlv6xr)Q|@$G1%sycQAt?-FQ2&Iug0}kR-'
    '=ea3SxtsAt~DkynOP=^@`6ovT{R@zRmytPp~PLymHh1+r?a6hSz4QleB*<torAsl93~96o^MA!2O=@c1zq&TYEZ`61E=6ABfsLL|'
    'TKUP(>k3zfu$@tgL8Er7Gmqvj<Urw^)(;}`}13CrNfDq%j?c+dQj(WOuf@map}uHC{dc9mZDh$xMtL^-'
    '64E<gzVNcGvGYD6~|?gxT)1{{O{Jv&}6Xy67tH&5D3<F=yyWqX^rU9VHSCr8Z4`G6Ti*u%x4otvNoHlH7ElL0kqxd!*oF-_Odz|$'
    '9JH@M(7ikXeM7SE|syHdYsgdk9HN&-beEFaK!BV%iP$hKKI+-'
    '7TqVz|voy=F5fick4+stp~_(=&sJ|HhrJQM5J&un|A))&s{8w7PE0SWXAXE3r38hE*z~SftClML79!=c^Iv4rq{gioGfAatVTwvY'
    'i~{hZY^xcq5F>RrRr~$0N?A4YK%edf5@%lc~=Jxlx?PIJ4!Msu3`g0k@41WK&8;+yR0Sx>g?M=a78so{Jk!cgnV<gSRfb!AX`&-'
    '2<Nj1ma=z=>|9HUXWQkeQ@qTeNU9zsD!vYbLzIr{XGFyCCOAohVq9BRN-8z5He*5Df!xm1X_jQSMq|8s{g-'
    'Zy&jn|hFn(EO)b2I?~x{Udtew25+o5VLfwNbvy_HlTq74aSTo9Wx!zSU1K7S0(&Cm|mbaMb%V@Qnq%hP@@F_qowlFc2B<2~-'
    ';ZPxJx(X!>^<)wKG33QA5(M#LjD4G117|xNL2gac{5f$2*P{(K2+v(Yn!Ce97>sR<5(%@RcJ_Xr&ffwRa$V$R-'
    'VFu)*@Q~Jqy56UR#}c{oiv|kYM6pZ+T}Bbv6-'
    '#u)a3)JiDj=6Y=J}a@mx;cV;(qQu;HRYC+`tYambye#ok$%6%!8rUbM=cCw4+PH-'
    '5!$4qo#)9TQ6WheW~oNq7uuP_|F`ibqrgm$|D9FvhuggV2LNOT2K{&veNZ3pXDC7kxC;O#KH-&9u41Eb22i7J@=f|M*L0-'
    '`$$C@d)Hykt0VC1YATgeKVj@RDMTS<P}||^88ik{{MY7dqdb3D9Yp8zcJ0;if|(nCCZEok()=ziJWqWNHP;dkpc`4RNrj6O7G#&-'
    'cW`QjxsWgvMQ@AF_<DaM}Q^6ZOLewMShf1K7tI33#wO&kPdYf;f_cu1t^^>R~+hP{BX+HACu7JnCToq;4V`f%RL6ABIueTgh&vgV'
    'EFo)F2F2?Yg5T&6kqa0k!xpBQ3Z;sx$qKYbHi{rLq(I%K`2FU(Y800C>Q5@?C6IO^QYst`vs?`2IP-'
    'I86!CQ<c4j6Rsmq&pu*X|42)w2vsN&Z4F&)n^EC5R(bsK})I<INF*DKv#ik^;lb`uwWbSOs=HlgGsL8?8BMZ#H+^z&p+LoD-'
    '<bb!1N@O0GKLGVPJZ2Vk6Hyolo;$U@`A#h9xYL~0P=eb2o_9&$a3i!SOZfJImicvsWqt${a8%}0BJr0`pECTI0{*G=D|edjixdEs'
    'W+?I5D;!=FnyPibM1^!AeJAtfj!@SB+uqmp$Z=y?{*{HE)k2U#G9vuuohkG-wx@eAAE<|gA;@D5#+Y`?u+<OVLI3wXmywl~nY=u_'
    'WMoO!PPf}qIV&@Ryu6=>_oIY(Sm|1cXxZlU@zzHjlhcMKpfx|<nyYdRK=M)pqQBMi3Pat2SaT}#*|2;D9>bp2o`(BMvni1#*hIhC'
    'YvCof<RkkWGXp=-#zu`uoWeuxtC6|5PSwKq(9v<)+Jt)UnP^S>)6(#PrY#urDE14`bfkM%Ra4ZeR$3`XorSuVtMx<U&6-'
    '7692k4(!h8(7nGZHZS6ltkp!AWZ)3I=zCL{FSm_KbOak+jT0yo@OkBJ&<tfw4jb4;cwL18<@zp8yNk7~%dpweK-QB)4tcXwy#h0c'
    '^@Wy;_szGs>)&+eC4{f-'
    'vMc>BEuOFmb!)|$C$?QsG0eq(5QFq?GO(U{kM5kPs{eih?;S}$ooQ6Q*s42M=8##UZ4%I#qYtwd1EM89)ewAXcvcl3^c-'
    'n7uHG!AhaYb$MSXiUp);D?Pi4%!$;1Dr^F=?-0XDBsR+<(S+W?7!Rv8T)j}dG6PqoOhKG8q0%NM+ojeOC@LHGxzR4HoX-'
    '5$Y(3IGSaNoH!pxO+Uwi9(&j!B!pNY;W-'
    '(T?iHN1g`bh*A>v$b|`)A!9u}tM`;U69KrlUZH!g;JCM=O1h;c~s>!G@7Wh1o1K292pcW{@%Tf{e=i29|Wf66}<(vLy<iovxayPT'
    '%y@?(}pt?122O-'
    'ra#9#!_`+t47s+Qky?p)5+}D2r8Y;)9)zaVfF=%<zbmBn0S)5xirYy>&V_{%~Y`7Nc?%^qaLdHZ#FhBM<%vJ_9DgVR9UI~%820)x'
    'FUZ1>W1YQ+NKEcae?g`-'
    ';UOUbnpT>3{q)m>(_CcL}Y7dH;OnYD$&@=e>j)7t3%&lo^=C^)m)A1I^`B@Q9|0h>~a`JwchwBCsgPMHBYMnArH?n=)J09)2CUhd'
    'y}np9T%r6@6dkpjbsKoAF^6S+Ppip^|nlv@iW7BdB@<f+IZEy9QH6a^QD1dFVny5OKVnU>Syug&3$C)z0HKQo;Cyb>Ca^c<7j*6i'
    'ax)0QKbRcG9Znl*AhuVyTjDR^S3`4X_}y+Irg3K`gkb|D=s2tT!FEgM7AcSJdN<~sFy5%P`%Yh-Xytonkqutnlm=)yPJ!^NyoG<O'
    'x1Wuqy5IzlX<-DBh)#Hq=8=FM<7}UQ0e}}`l#^29utszG3b7Hh(Pnk{B$bTHqu%w-'
    '&YX16{%R8eT0hrR)ICQwn7ssA9wXpUk)7}3%6Q{DD;YCWagG_eMbDY)pU#&7peq*`0a0d%e_h?#`=9)b8fGZ1(t$RrXX&|lTygY-'
    'v;jxII=Qzr{R+w8}-+24@k3Z=pFPxJ*2U?tu-OlTFPoJ=_?a#+GM9_Pb&q8jDJ?^oKIZUu{=8sY2I;}t!hlIYB+v8DX1?m(U>Myf'
    '=m>7zU_VA5xcEbG>Am+M}DZS=Vr182`Cp{1^3t*)J38iOBs{G52tG0I@IycVkMtl>**W&l~{9zwpBaZY7E#7Q`e|#YJ?O=x;Tp~@'
    'cZ{)|KZsPns1R#rQ&vCm(M9*X|*XOTQb$lS<r1dAf*-Dv_|T+KE;B`=4~nK{6lpdzi?NuuA#1^jof#0A8gLN<tv?I+-'
    'vOv*jzux$y)Js@pj%gp)s(k)BKZCwN`XWz4bDbH`HOoTIbKHqp1?--'
    '^PtpFFTR<j~xVmL=$gQ7@k%yR@<Nq&Qq{JGS?nemmwH3pLD&P1r%sap`B$9XK%gEB)7xQrvuXPs5cpZ&GnRZ{wUjen|}yi%k(w-'
    'oyK-eS4R6>iF#gplx}Qr9F>pu1j0TcX~!n4B^?WdyOu$r+v*3Kxu2Em`NlCZxawxS8DSgBZwkTc*HmN1LHRy1%U1QLY|hEZDw*wP'
    's>VEt+a~0eo@jQko2A?^Alx!7#5Bim?zwd2nML+yHH~z&n-'
    'jYWp|6Fow7WETq1VAY=lj1af)hGG0dkjZ?Mj>~#ecos!}`%0f`FKEYzgdfRi-'
    'm(>16iaovfKtiKB7Q$A(n(iS1npuOylq)0U4XW~jGKrm17}U8AHX*4{p!-BbWlDL1h9LMIns#-'
    '8Z9cWbQXxND+L6T3DJ1}yy#%eaB-BC0W1Z>)f+5!YF3Jeyqrw(1O*63idCXD$1SbtHmuT|k_h1Ig-'
    '*QUGI3W@8L`X}m{@0h!SyO_^(b#%djR%$c7}At88zAw#P@#nHF}25e5_;y`p4XXKh@YdH6Z2Aj+sxp{_!hZ@Lv-KwJ@NNR1zVBK;'
    'wh&sD?p)Lp8L^b?uCRnT;(0PUF?K3n<x8}NvGn75G{nN_Q6=_#0WoYGQ<(n(zjdqW}MRikGtJgMlKy*_Wn&fF88Q0sPuVy=}2}SE'
    'z>!dSR9P2Mwhl2XKoZT_rV3viDJ(6m+o|pJ-UDI<b(`zhCt#){?pHsQozkxaSP@RB1ZQz48fGlPksl*@|@BPU({dBM;9`Bo5du$O'
    '?hx}g67&DI*|MyHl(zIQlw`Rv*ZA0aWOKI=#*=+b{V-CBoZf|+0W*Uu@G?}J0uN~y&9!b-'
    '18X){TRaif^CK`97wpa#xvkM%*b<8#(LTjB7P2tKy3-t`Cnmbycv|U+-'
    'wASVL(NYd`)wJ06w3;$bdpE+Jk9SVZVp%KZQS|1r6pCWi&bPCtNp+5WQ*YG(#<@eJj44V3ZPbjhxi7ooyV9RJhLF+~vm4agmj*Me'
    '3@L%OIdwd+@7G?f*6Xd(M3%wOZrO2<L1C88+0?hGieI;u4Tjs#asy`PUK2}e3j8B2Z5?Q?)*6msygFx%yb4<S?pZFDTM6EUttUC8'
    '2{8BUwxbK1iobHXC9+9=)j8uLSc{n0sW64zy8bgI^cm_3Xb-c`0i2Za$-zo}`vk0Ro8xLDn6~ylTLVW&#C2p~dDZ=-'
    '&RJjmye$!@6%FeL&e(tzGxhboYnNwqE!yyAQtvuiN;ZBY$8WeEtvwD5aZ>Lz+m1d24e+gXCJZXK%|x-'
    '*VCkzi4+H5ae7c%be=1Ap8|bVr#@-'
    '8qe@_qVTPDnRxH0Ce414FBS|ej#nP~6U(=qS9e#tfFJli@Qa+}bopEnhk_r%hc1PbQXGh;TI>%t;!7Id1Xof+Q6XuFolEXur*^_%'
    '-_se5#}(xif3_w1WeT0+iAZ?t1+-fwFVao63lo&`<uIFU+SF6AYqq38J}rJ?TmCDUq8#%bc{*-aPC+Y8wJL5-'
    'a$OD$)=o9?SS#`FXwm)y5rt2I?QovbqSWmataYwHl%WaW4polTVMqccljcp^Y^t^fG@umA4h%yHqA@|$0}AldUXmA4zV(eW`}rQU'
    'Sb$PO=R?4Dk`#!s~st3)tt^zva{J3Wu(VZ2Lm-#Q73VdZ?e*T3whLHV=+LagIr*ttY+-'
    'e2nnLrTdvf5n@={qXLm@816S(}#~AKK}CK`?v2tzWw(PKm8{@<=mU(79X|{<}_{2``6Q(B|iP(@b7f`HzfH%{&4vZpI@Kfc(<-'
    'gTzy#6>ZkmM>GZ29gy?;km&MP)&*$IxH+UuDW8V9DTh`~t{00x6BYxpLKAHTQ;_KhIH<O=Z3~pP}vQBB4<$XT6ApgWPJGV^Bf-'
    'jgik6$m#8s{{J+2z+lz}Ka?Z1d(q*yOWuaWQSvGH-L7+$39x9$&wK0?U$O^sA6yN-Ms{N4Lyri+Lq+n(-'
    'FA3E#FwabRA(|MB~uKYaK8)mN|9ZJj2cCY-{V*gwDh_LsNcz57LuXnh0vEYq}x7{WBC<+N`fFX%k}FP0ZJ>@=*45Mjag@sf+ttv-'
    'bbTaHtZ?I)Mqn%unMP}WSOX-QjhQ*drh&aVM4t!~=p36#VCXTPrbY_>J#w|XbvFK0cckU(iCr?yR-myhDdWlbrC4WD<LlaIOy)S9'
    '9!Vp-'
    '*6!NxP*Ib(I(42mS@F3I6<w&%hV`%V7DMC&|zH{*P#D2uev2ZuS&bHd)SHJm^u5V&`8n+M}<!j73jSo@OZ2`|l%4wo5z4xmW{X~8'
    'j>{qZsE3H$=Scf!W_<O*&BOQ&fD7f*BMz;zW40zEf;!8~UQ&G_t$LxAXEHJL)-@Cg3cRxi?#o#XgsnkNX*hK+)@bG#yvfkw-'
    ';P7zymShIJOJP@-D3-'
    'UPYl;rf+Ofly}6d9Va$tmX7T!I8REGsNAPMI?xDJ$NzF00Tj^Av=0*<4&A)_}!lrLrwc+?EL9C)r^p7)~QWT0_E{Mcze>mrV$WCx'
    '>avVxqDXh?uspfvnL5A7dt%-_}{kEpi|yObM0>f<-'
    '(cKrJjdmiR29HIhMT%JG0mb8yo%pO_4KZ<!!tkd9q%xOKuS3Hk|(56jMESRZ~46E-'
    '#Bi)2AUE(jq02%tgM#vnxk<!9^(M2MMYi^zH0z^|ev<y-@_85E08X_5m=yBw~d{Sxq3{yVG9DS;87Uh(0qtw6N&CWWlOg`dHD-'
    '9!s3wvc6J#dl7jA&O~}eS;*j1uS;wGjPxhWd#k3h4@`Eu_q94$u=Zpc3CE9vb2J}Tf$Ri1s>*M-'
    'M|&XfuiVyY9dBhk_R)wXk>bWbyksH2pWzdh#cYrQ2JQQHbZV>79bqG1N(~_1Y7PJ)5i%Pir_Ik%ZGeRn?Uo)<K5W7w66Y2eYq<dg'
    'eQe|LA4pm8VY%V;tL?%I{6XloTFGqXtf|(Ib^CqtuL^}a^zFYJBKFPR<Z6vojI?~&!D99paoRUx~^F~7wiXe8H8~7dnX#W8mci&L'
    'gFM$3UUP)V=tN^K?e)X@PTa;riHc=gDNz`mx@aOA%-'
    'npmlMgr#1L5U^fsS~w3E!MBwL^UYd|_xk}WVc4%`L}w5E`4q;Sdv;|15k`zg-'
    '5A2QDl|A~*o9mq`O#r>E>u%V!$6Iv`{OTqV0IuPWP^%4jTZ68EpA&*h^hBd>}*u<DXx$hb+iqQBCS|1F*W|F~<LeatI%vc1D79?B'
    'VfLUQOH!z@sGM{2E&1$}uPBPde50wUM40(c3?+Oiy){C9o(!6AS=64E0>wv=G;+@Nc)R5tjAQV9s0Rs)9;0qx-cwc5Rd<FJ0$wsl'
    'sb*Gq^1##U(j`9|xC^(!z0NL=BJ`>S{1>w$!S9wwg_yTbNvp1Ho^j$EaA6C)Ky9amL-K|%VX_vK|dnT8IzLmv5oUoZG#u0LfvH~-'
    'O4UR?J84P>?rhsjNjhy|i$>3BjAOv;;j!D^>f;EL^7Q+BdwuJ15dvRBA@}lhL6MuoB!G}=c@VVCPy+WBl@S*#$Y%kvx#r{CoM^TF'
    'Kk}|bK|G_;0O%q&#CG!{Lds{-*{64K0#JbX9*|9rQ>+JmpxN~rlVB8@7KE(lZ>7hX3vBUX-'
    'E4XYsci}zaX+z<_jXiM}j?z!CQW45I3*QXi0N$u@)&w%mnY$oV0e-'
    '#k5%lV=!bEj1nVZrgw1E&JVgO67Q1xrfXAcrWpTkv4h=pSQAe1F!8|(n{7;%`Hw<JUeUi2FHa!ECM&Cm4mR|K2qO*})$2N*fc4T3'
    '>(Knbh@tSIUzQwMB<6W!nvM)8icmp};^P6sIhqu^lj^Wk42+Jj>Oi;9?FXD>N;1UC~1x5o8kC!t|Q(qRwg?3dt(M74|Z%3h7^c!u'
    '=@Gf&`k(VUrJAm<E9d*Ms?e3`!_P7;FxT?g)dlE6K_ymZ>bS#@1gYz4Q2JvZ1$1Q^#G=0PAFm*`{cRD!D0%kVHG4i*{%La~Rg%B#'
    'Egk3B-!1~^WuTX%B+1+~IM2*R4~Y(f_yvIJ9Wc98<~;UvZY{DD6=d3F-jy@G+@-iS@e-'
    'r6Qn7J@Rk;1Im9t1@v_WG!$&!X7nA2<FAn7r)G9M^CIeOy!L8ck)IMbVb+!Nr1Bq`2eBg6+tf*TnuMgUPSBxA=@UdIv5_os60qxy'
    '&4G%S5Lxocu#YFBN)x?t{b>EiB=Zh*6+ci2$Se)Xeh)6nXRBRvY#SKd~;co3DGAJOsGM`iF=3u3Mcsa5CZYJw72D$3PR=O1mG3#V'
    'F^?P6l#LAli+GjqVP6xA`nomaJdi$yDR!l>!q8TH2M+5qoM0|0!}!_AhDiAdUHI*=vli%b|n0s_SY6zGcW@V8IIa1a8D8=ie-'
    'r6fn|DuAb1YE5U4o_WY+v%cmvu6{urFEOhQ-'
    '|;ZfLYiN7<y;Ym>}0gMm6Lq3_kGH59IKKYIvD?vbl|3)B+5F&fN8?+l(M;Hhvii4?=Fzd&BuGcImPNx$Y@S<Ah3I`UXIx!}AZ;OJ'
    '$h=lFbg1-'
    'drLTCkbiPN8V9>gXlIVr@hepl&Owh#~;G0XC^rxC=0K?_b2K&0Jv!mX2t9l{4ww8va{zr^7<5Q*Y4$BC;0+QA3*;+)7!FoO#`;BD'
    'FYBR?}p5DzjrSh<wFx(#MTUKXqhG1yXP!e`oD5m3$?$|;F|>KN12sdyva>Su4@10l2omsmD9VG~bc4{i}E!4Je{u>BJt0IEXj1Ob'
    'HjW^5-4W$<Z4c#zX<M8Kc$#=*HO$O9FS2@QuwxD=We$FXFp=3oWK1qXE9J-Xt{epAMkAhZL`+$6RTE}iV-'
    'n$qcp(xtM5y%Z!TZKCGqr6u1Tv@K#8p{~Q5@Wv$QpdeydIUvLnyAuTXIN?*?OZbj>1u8UTqnO|tft3Vd5<Z@Hdq2T%fdLkLYj+-'
    'DiKZ379N2mi*%AJS9>7xZVd$R|7h`d=fT-IdUPV^ls}p}S2}}xycioALI6yQkSlJ2cMcKF{t{1U@&48hR9!eJ!bYdBT95S3f2y)X'
    'd6(104!a5^<2vA>6<ao!fV0-3W6}k!<DqkRSDe*aonMsLr5l%=#a<E}0C0s~+*CHQ}CmZC&i-'
    'PZzl3)T^LBrw;1Pxh$UOHi3Vmbw|f!H7b3_C3Y9t-'
    '9M|7TzB%C6sPgtf3r+X{z62qNAp7@iSRiHs}pwTDLovx~SGb^<}%8q}~FPNSIh%v{i~+p^R+SAL1m7rPb|eA>y4ZHtE^V7|*pgvP'
    'muFa#$SmO(s}L9Zx&I6|BVH4SM<Sq$KcxG1JTRMoyV(dSU}q6Q&2aVIzIAgmXxD;(9;ty}|_^j=_R76A?f^0@dw8E{vreap!zvC9'
    'Ym8-'
    'f!LBaJ8qE;&BE6B`y8&y`Js;`x11q)TiT7;;M49|T_t)$m1fg^^u*aHSJuG6=AB0*uxO^*(z6ykRkSmlca!L3+3rY0K&kN)6#RLa'
    '_Yg9R$FU!PS-6HVD+XlUn?}RUlt6(>_oYtdUxrBGKJY@G<#tLF<BHtHjn|kkY_m1F#eD7oiJgxF*vMcq2B1zY-zd6Bd#E0fdv{hj'
    '=(eqDV9EIXTJfJWb+s2-'
    'O5IirbRogC|Xz@JP7^0q2RZ?D3V`5fGR#mp3l@93K%E8Ii~i8Ek^DL%gSa7m*<~C)@yM2rkVcPFlJOu&zMG3U#X?d_f&EBbQfy6+'
    'RA=m@zCG!XEZuLR1A@Uzk7e3+z%3gP_}jpd^TA147Pu*Q=nIKuciuW!_^4IATkJCcyjFuyZnWhWJ?^qG$-'
    '8?_+QRu$t3JS!ETJ;7U4CT!>k`who~+2!MU#tD(>qF&J00F8to|vJ^T4{EC=MpscVhby{CUxSMDMxt@wxo6`ai%_jzRy?B>mnZ;{'
    ';;^ZVDJva@L9!}I<5HLmPj8#Y*^CgprGd`FkT7!nkHw*~&ouDn)0XC3Mpp5Kf5hO&MtQ-'
    '}~oho#|ya<pIvWA}})=YAlns0@)m}jRiTnTI!tRwK%ymI=6(k+H6MqC5`aSrj4lP!w1?p~CrV7}?Y7pwx+!S=v2$eR*)9o|5`-'
    '@<Y1C=$#foL3M%%AIqLy<!v<4pGNur+hUYC030vh1fva6-'
    'B=9fh!;;5R8{0FK7a=Gad|`uw|eRaRhcPehQc(>^`FyHJlB8;U%<jJ8^L$TLx=ni)ZGm5VwqbMY>>9#g#&Q9WE~V1t^232+@Flfx'
    'XqKt2h`dj0r?fFmAYFAy%jj?-MK~*TjOX<^T^`7b-'
    'G|qlS~3^ZTI5Ldf@CUW{4V9o!&xXO?R;$VnzPjMFNSp<Hil+bLp!U$Kc<1y5!-'
    'MAQ;iH%pYV$cYNMA&7#=%5^qKzT=gdy|S+<xbwP`TY_&d;70J`@6^bhF;{nwybAmVG|#&g!E#A?MOPJ2jjJGl*eoRV6pO6TLxNYY'
    'a`Suwkl`T&xe*udW6rD$y(|bUTm>ASgv;t{Lf=_ZE_UlGH=#LzUL-'
    'P=JACo$vQHtfm|VV1OO7Bjpq@m+@NAKnEu6>;9WEgd0;&ine6Eo<R`OO5NUn%vL}F$z(M;%D<>u1UEA9g{z(@3^tR0#HY6j*htL8'
    '~~NHPSre1+@wBp12Zi1-'
    'S*b)5H%iy+Iwi=u=><}1)(iP?=^B*;BEC|JoNUTT71?l@r(_#bvl;C}onC%MRNsGw#+E(=$`XOV~!p9ONA5hb0-'
    'jjtA{+zFHtuPk4@beBVTx=p#HcRgDJMsX3pP;5_-'
    '16l35xjd@!IT?8buZIkvLIt|*6dOXMy##nM<3)nj<4}@>NGU^ZId=n|pCr|Q!r;iboPrZkEm(>Z9@;OLK`szYU?6Z6{FHnuVtw(A'
    'tEjw8Z16AydJ-ipsV|w>c#~WYibn`{XeT!KBZ#*RJ2+2QtbOS&&zCpy&bJ(iTW+mfr<QM2{1#n}z=@ji(h7wPb-CZ$z(iu(Vu(Pk'
    'd1p<Sb|=@ZYZQ|*=Lp-$I$-'
    'exz1+lL$_5v?9u4BL;2B`QOldKN;8z@PlAE9$2O`uE&>Tt*kKgmxf;h?8i7a%_^notl<eTuap!6rk#!AJdMaaHDA6?-'
    '?p|}BiCU=c_iO*3bc*)la!Apcg5U*8EXp*b`MQm2)Y`9%=S0Q+U-'
    '+^QBJ{(skQnJWhY|ffULIp3$JYd;)3PWZ=ene})4EV6qdl5UF^7XUNw;cN=rriGHE1>LTAB$ui$W>X+zDwCa!x2KjAXgRQ{>+lF`'
    '0@R_A6|X+>W6oK{>!^R{f{62^zN7UKfU_v>sLQ~_>b>DzWVCFUcLLbkKcdzvt(AXtjkuxtQ0+e{fFQE!`G4$`FDSOEC*a%Hcv0-'
    'UZl!|jCxLYt+4uk{_F34|MzdP{(t$`e|q4p9Slu-Qs*vf?KaQd&z?TjXz^`sA?|DIx?R92Ggf|k_4HeM-'
    'Dws7{J$O^{!v@UO>a0~$*?b%b@Pg;sjIy||Mda+S`$-yXZJRzo*#Lpl{#lm={c_rFQ~9@PZ{k~Pw-t-'
    'wK|8s0^Mt#tIR=BGOIa@UF(x7qwhr3Nvh41YG$n0n>)-SL-sTx8%7B^(l?{5v-'
    'zgkmR*rYbDgTSfybkz9l%XDO)E#VWbCHRJ)2zjd?M8|>MQVWG-'
    'I>s5{uQdbXFp55bRXDm1zs{=;%1>mc3z$vetTxwdQK~`6)C;zbUYE7$O8OM$A3awKk<{aJ+iwa6J2VEgdNy*<A614BMRjRvj%kGK'
    'PRo_a>bhxz_J+qpXM2JHp7+T<cOd8yg9$o}hV-ZcmT@gKIK&PwZ~#T-'
    '$kr1b2I4$Ka9YaeN0i+2{gQEeGelt9rcMa}{*ov##4$w}s|esTM88Z+|z^M3$*i<%t7M*2os=IMf#Bu)Ye<TPmCMjPFT%V!0`G3U'
    'F&P#4+pJ%g8Ydt(c7+P?pR?C>s-(HNMJb>cbD17An5^*=f4-7t<wo(BHCTr47-'
    'qYQ@7GGxE*K)Yf~@*z&cuR*80Jt=1^>zOI_tB{U)~FX+fwA(-'
    'uDUAhh+^rd~hb}V#vU#|^*3^{vj*je{3T?_meLIkLH)dun`8)1f=_CY_k`Yw3rz_?0F<a@YPH*3jn$1Po#Ti7`syJM_79STjlnNb'
    '_WAZcaHLdT%*rDN#UbIetnB6lD6QSq3%Ab+U7fBNI_qweCIF+cT?Z&r5yx|0LUx?FuJUp(SVn!UUGdU01STEm@Fv5WTjyiX3%x&O'
    'v)oH}N{0i=HtU$5-QcUP#-<o$KCYQ-N^43KSj%k@0C5fJX~-'
    'XX(}40j6gPEg_txr_8Z=!^3fDI_;?Gf5E^<}UW6DiFlFy=LMDGFR!7yMav-'
    '4f=d{`CgmPXVX&ma|MmC*WWzEJK%p#^V2>nBzVXtxeFY4azp|dxASSPdIyO5@~$P#Cwp-'
    'Rl9A*dej}e(_+t2>si*#Z5IM_y*OJk$LHb$GOEzRG);6ci^Xy2_Q6_9x^KvvmRT+I)8Jlpi>8pu_IzbG%idv-bwA70yRPNc8H&)j'
    '>ki=M@+1Z#uGMFQ#Pp~MqdvWjoFfbT`C8F`s-'
    'vgesDgUi}r7>5nP1BOydA{hekuemwt`5;0OcP*3Iqm+j=?TX+!S0vuvkBtlmkCJK@c^dI;|GG4>PvSEU;4II&6Dl*q@TXXu`!gN_'
    'jbKixbtgBle|xL&&xU0^9woEkCyS>FXd8~B~U|%`WR(e(-~Q%)=?Tg>%-fsIX)-'
    '){?C1`%xdKAHT@1_yrD>e1|7DXs=k~5p$+lKp<eoRr|mrRpHe7yrwwT{fLS{4fU9d8fMvU)u-'
    '67%7LHsM%h?hyyNGy>+|K7@Ne(;;?Ni6&0Lag-EZD<)HmEr7<oZh4pD{<IV_J1*?r04xy;nx&r~m%@W|Ue64Bg4|O`qBEU5+BLYJ'
    'sW=ZG5RRphLi9PslLpST>wm#b6%woD_Jj!_hSq*OCh2t5L^F(L8<neQbcT{l{WSwBWDD;ZXpqrjvf)euAh0eqC<|<(NsQbGh%3gz'
    '(ro&AJ&f4mg^&&63h%JIJZzde-tHhN#JkUup8q4H`zqMVcpgQYlj6kSan}QlbIcC{HH22=!^A_wKepb46|z))iy0t~)(dMxEaIYv'
    '%5BluTZ*MF~07Qwc_)|DGDOyC)^Fhu3`8KHY7vr8<=Fqdd7xXz6^iEdk@c6<+g^G=>**(obvp{nvl^`+Ioo49XkI4{lYfU0(i%T9'
    '>ZZ%6#6t;_pVt+fIRaIH9H5Pm0AW_lyzRxg?!n{NWIyo(HW`Y2@6T)oEb}T7xcwzaQy#L^~@e@qwE5T7BD?^UN}781Ib5Czc!f*E'
    'GA}CV6Ds5Zd}{l{ls0rP0F0lFDb(%pTuPm$8Tt0Z<M15}j1B`=;BNZ_1Da7R2-'
    'uwWnQ^CfN^$tmq+9j^yyru;a{(W#^|Z)^lU6*%W}mlsj>xFPC9mS`cDEu@5PY;fFr7o1YzM&Png;6mDi9|2<F5wl0l#iXZ5ml)dk'
    'y>!|<jTcEKmn|-0hv9F(C?LCLBEbw)+ZtD{UpW{*Jfv(HuFwfs1PF2%N57m<P(9sxdp3T?DRg^jYd@8p1-'
    'S7YRi4k8sB2}|{^{XDs&IoRMnb}iC4F==eB4FfRh9TRk+86Ks1t;1SmcA$j&Og7zve<6W2Euf%Cq<$2aE!@b1E_@d#$%k=xmjgNM'
    'Oq)rqa!)?*86o+wkYxAuP$|~u*FU85@R|gKr)!<)*;!ih;NePYE=!-'
    '?8!nWm1UosAM>ghdpuC9a#3QelCRp?s|rriy=VqPC=9JEinb*xoxi@SBi8{<G;ANs-z+5ImO;EJP-@WHlvG;scGG-'
    '6XEBsoswS9?rIut?41f)O($T!<mu_=E?y%OTSJzrFR~|*A+FCi~=u!fSgZa9ZU7>>>H{4yl54Eh<h~-'
    'ie(%ahGLdnrIAN3(E2DM>?BRQxPr%|=&1BsW5Gp(V4V=PD)%Mw1yS8kM2-'
    ')SnoMoV_@oc5|cgrXa)MVkTSGF)n?D8!x8UEOjg=%D1OW7mIoL;))9usX<PH0Zy)vUp)s<01ixCJv}C7EqS*Z3SR|>;~T54aVWA&'
    '9&^^?)DEabDiHfHol(LCZF+;uPgD}ORZZ;<z*4FJ^!Pe+0_@*m%-m9Kk82Q-'
    '~aBn6+~(@2cwN`C=>g}036qXrIe$pSfS>&*xZSMI`!wn(lWzo=D_3a*_~Hv^w~%9*A)(LG$BzY7PMT`{tjPr*G06_poE6>S{nV6c'
    'w7@bAH|(1&Hen+ExTTk9hI~-'
    'AQ`#Tvt@hG@;Hiauu8|A2OcUjLl2(~t!8N*4Nc)^E|+>M$ES@_50(bH+&q_i)D(lQ@wIg_rAaGUx<WQr%gx1L7e<<3R<8Q2X$VZ('
    'sQNvgD??JlG^I4q#=i|Tu)(;BZl{SpRI#YDaQe=lfg&1vQ7nW7+_u?pwrl*Y2ZE+V<w=dKlHpjxB1>&V18=hREHif2sZCODKzF2r'
    'u8oR^CVw>dxg1LOI0laRFA^M~x6{80bma0y7E%u#+I0WCmS#pycSW;31?C{E(8G9&W7FE0nwCVS)84-'
    'ojtY{Te`qDJJo03o7e`MX3~3Lh+7EkTNdFNy3rCf-'
    'Z`Fs`k3_jW>H!RiXJL#tXyr&xvfDs+a)v9B@@dVlG{jW*Yu;$@N~+6VDI3=T?V*BpWxYx>QvKw)bKRlpY_O5yKPr-'
    ';H$c@savMtx2A0&pZ3>4O#yeJmH2??lfJys_<;4tM%d!xdrz|L$j-'
    '5fqdCW^s2jW6j*%agQmX?#61)G;cX~+#{8?!n$sJ3@0dlpeT`-'
    'I$A)0n)krpwU%!B%Q|l@VA3hga7)MOAwvJa3<)OJs7Cy|)7wV#Dm$%O5VcTnD1cb5cvxZ3U{`eIF>cpk<cYa|P#bI&(K#=AQLqW9'
    '>Uxy_~U@>4w>A#m{qB`xxbdN9!tJ;<erueVPjMhm%&GzG$h?hae38Gr~3V7XsJxv^V~o(2Vj$nV<df-EN-'
    '<q#4Xw;qIX6r<YG)(fTyql#zG$oon5PpOk^x#N=p&Pd$~S3GRGduck0G7&#8#Fw{nc46N;}?~wR%m!l~#2z%EQ+VjCs+;IGK__b@'
    'JtbtS}-5&7>-RW((;#r(uD!Zew3!D-'
    'sq!~wDQ8vTNcr!xU_=4pqn+q{$el{x{QEqn0Z8_^aS;@y37{#rLO#`)}?Z`_l)Pq#AXtGq%EDxc=BnBW}m)#MlsFn53hI6tM4yBf'
    'k&6BzpXjsB8#*9CnwnA?opPOJ>Q-PQO(kA!K%*yVR3$=dkcqO6|*m3-'
    '>8pv{VWlRAw6+=^QY^TKF%tPyDg*6>a(mcKH+9KAlgw7<jjbN~$i*(h-G?+u_d~D@>c7-'
    '`tx7tc=Finb$y5iod0!CGF@iA^cHl{V+2aw@|*mYm`#nWf(H^?lM{-'
    'kh4=h_DGn$Mg#V_$N~Uk0y_#`e8QQ<tKb4!0z+kEgOV^0fAlxoS_c4fY@*2+R2uOV*QlCU4KCTvtPJJuqMdrqxspJyU7EaJ7Cv*='
    'qpGohjaX3xO%m?nu^9yZ3t7I=XMsjGmSdH9Ow4<6v4Fw%QTR(>{8dBq%g5<&DJ6YL1N6@hmr>E!PH!&>{?$8&EB;VW$$)NC|=BI+'
    'nyXH8Zh~Q&TPrSv^MecD4m4SNXliG(2-tBbusYVU^_q9ajw{PjOB9B%qfK()WF(b~QmWxM&t>K4Y&efq+Hz`m+z$A6x=JdO4mp-'
    'Ov^}_B9rok2S8OZF2d}jj>}pU@BDPiL-fU(+(U;skrC~;%P>MFKZ9|<!k@)$A9?xU;bE59^e6J-'
    '!$q2huFF6%OS?&o2eX~(u;ZcG)p#p27j$n5vM}w`Jr)$HiXgED_c9moo)moj$N%%3GZoe7#Sk_pbtrBEp%;R^C*Y$GO*60xYqpii'
    '7!>E<=kkKo7ZOI4ntIpWgQ)J_z4lJ-koe;AVT%Rr*#CN8ftVifKOGL8XrthX_ieZ?yl{|*gghOx%ML%R7X5PQ*BJ?KjR2Z2U{q_W'
    '3=v5Er;id&K)q0a?j7M$60oSLLL%&v`?r+W(b0zpd5;iQ*!^c(H*%X>BVETHQa8i1AC?^S^RrOc4zJ>O7It&qs8u?DONVxZHCWQd'
    'N2$%Whc<zo5&YY!rakl!Pr#61_5f#qw2B2(_&xWW8}TIk7_#{!bNP$2gE?RGdER0H9B%L_>(qS9T#6(1*rV8!-'
    'SiVg8K}PZUS{rP0va@!4pqj>L4CXkgbv;+S@+GFag?_)L{19jYX%uNjfZb#NK(SoJpou5jy(~5<?e_dD8R*&oI`Rhpp$=+)L;IYf'
    '2frg}_=nF*o<^XfjuyYIF%~#3~8aaqx!-XX7W*pND+XnBH~ZuzBt%f!%0r9I=eev(!C_aNi88$$)yb$;n4^@X(^Vs@n-'
    '&0}Jh)Jp9<-6X)fiulWwJPzU}n0)!P-'
    'sM)smrB866cazGhy;HI7T1BUWc>`4hys?qJn(lEJ$r#uu&5$E$UW&0*0ML2{O(o*wJE+ersn%)Lem3{NKH5h5B3iXJmi9wgUdqul'
    'S~@=0d>!J=o1SuWfh%ZjZhd7^<nejYq@t+gXH(5qsi$77X|?7U+4IC$3`319JJ4jenJq|?DjSC1DL3q;+ToqWjo^``@<@SxwV#x?'
    '(Rw0F5hP1-$XDizB7FxNi}0UHxXa$et9Gl>J?BkLI=0_rabUwx#z&lwR@$7cYZyh(+Q@fb@hH)WMlkfDnow-'
    'N`z!TjS*l2Job%hYffDUBW*!yalrpwOjHRF`8IkP~(pH}7D8a0Dwkq>fwKPFed{$O<aiuui;aU|pMtfx=wi2|u43p(4cs^?cZK5Q'
    'r5+)$|;3^;MJ-rl2Myqsd1Fh7)b*#0djH!>3?fh*D;v`a+dB{ael$*@Vdo^XM^<?tXrzrI6#vXpXX;cbXyLUGTjVX=7?w47`hkA5'
    'x@gBqL8~Yu|vHF}S(!dOER%+6hO2t}0-'
    'b|sn!%o?}C?)JBx71xQQaMMbQoTYi(jTp5q$8pDvG1f6I<LnQvuLUn^W^pR!HhMVt+QB&O(8#)3rBkmjQ~0aQj}V}pmbldLU+bgE'
    '+RY~Y-E}JZ66I_yaZH8!ZuZ_0WL*A(MB{n;pkQCjC*FubauOG+l3aEaF<jQOA~ccwrXokZ-'
    'O{VWVxp`QnkMvU#<CWs3R^Pnrf%C6H|1z`r_k5UFhoBFntxSO>I!-'
    '*gVH?EH|W4Y;*#C?9FUC5aC89UQZK4Eh|N74c3}?U&T(#1L19^DBegnrXX>KYz*H=Zw5p1PE*R;ZYra0kfA;{=XV1QOx@Jl@CK6U'
    '!*kkM&*s6NhAgALwUywXA^?VTT~kJ1$0_%ZbT6lT7OzxOaf;!1FOAsrj#RrB5b!H5h;!}0vUhCLr>~GJgT~JS?Qjb4_D=Qy{ENVL'
    'Xq0Dd<n&@GL7iBRJ$tXb1hn+hG?qX#IYal)X%2>QYpj>k+Q@w=wC#%kI&Ahl>GkwD(7$UC=V5hq=DM}YXud<EB8G;~R>rUo%@AO>'
    '`}Sx(w)df6ZXFA@PUjlQQ6MgjaXPzYycubdFxWfab*0`|V*)lO%YdESCwI-(k)}17kFDFDDMCXCT4h-'
    '2iO$mK3LEx9MWTEbJa1dprE@x_{INpHMqpi^er_CBXMhe|=xH1N1PIxBS`B1_Sip{s9AE}9&`v!xi_3<=+q-'
    'dhXg)eB)!69Ds3~vgGm7M>V@2(a4%r<qOD5^4tu=O8H$syQjBK>CVF|@jf81KiX{?WVMqmtV+n;-r-Tp{(;4gjt^$@btUHEkb1c-'
    'y<wnm%(z9g)3w9k>krpJ(*_Lx=o*^ry&Q2VPvp!6FGP_x8DEYZ*>DfTjT+ir5PR<U4joP3XwT)gh9{J}Cl?L`ks@f&O0Hga^WD1H'
    '1K7{Fm2IM-K&W-T)`)Ob_y`!Z)W4pn1mP9EvA)NxB}%L}19Fx6(U)I3^`Xt<*sY5-'
    'OAt>TU)z4@oU;kA4y8DJyJS2Ojs*;5bg7`(PrQ-h45o?(qjHyw#I1FO^vwzN_*h(L);M_Zkyv-'
    '?#{A+q~pQ;`#oQw*2(q%4cI&M&y^oZg_>5&gh!D}JoWt-C9z6x42&0%LWgl)cWj4C9mtQ>o6h<#6dmkUYobiZ-'
    '<^_Eb<efHu=*iUEnGu-'
    'Y@*@cd+xT<}(dUBn)ia@J~$j2me%MEhp0??S5N3TES#Dm~kDJ5xuvv5cZ8D23J689M{nhw0LUtfDOK8IwFOvNn6bD1hgKur}(qV5'
    '!WiSehm^aZGLZFV$ar{#%9DuM$A|j9EZmT+UB3CbkTBYx=?KH@}uLFC|Zgpe%-'
    's*^}DlPoFiUBG(=129`l;Wha{W&0Uj=YzbYqq>ko5ruYfUz0?8$^|o9=p3GgJ!Z(K!gG~3^l~kZUlcF_=xAuLy>S?;%rJ!T5Foij'
    '(JvCJwO><1iMQTqt)@iJQ*H&X8;{_eQ_MiS}D`+zI=i%*%D3W0ULdS+j`XUeA4>kO@mV)YP#nAu1DNlZCAyudC9l@?h&Xc+y9=vG'
    '3t+BDGkre^xwnM$uN?5sZ&ibz#RYf!=ZCxLkM^W06(`=>oo!#(s0kho@wLxs#2(g+rI=JGF*6dU*C<r3Y9c^beq{^L@vjT-IC67;'
    'VD9`5VTHG*P%mxgwL65cpDzu5R3D`IUW)f(SC-'
    '@K=9<8pX6jZpgItQ+KFVRHT_)u6=11r`Ez0qBcv@*Jsn>3y%MSuw2mV^LR3U1K4QTbfx6&a{)PEPsU*-s<r)xcb{-'
    'i@s<mFsl(UPS$cAO?{8R)E}HTbg%P9Z*$icc8+HHf(2&iAd|p^5nIq!F^Xz6o|6-'
    'XKvIubo6BT6w~2o&v9~1$Vv*U9fG?ghz<s3PQ0OUyw*D14nXPOWM;d%cWgUE(k_<jiW1MG?h51HA$3tb+-'
    'a!Ux;2AdkBuv5*kGOZF-}L&4K^l>w%jshC-'
    'h%n97{85P773zs8u5N(1u+tbxMmNN^MRFnEU#A+ZtMNZm!k8*fw8eGZo{6c&7zTF!f@>#@SknR%z|HQ5t)<R3kQ<j-'
    '<FD<6SRpCXj%m9OP@DhBOVM6vYD=q45r(1z0t-'
    'hb^+cB=u%jd&bL4Q(dKf$h@s2v(|>sSgT8sPP*jY)@vQ9L0X<8UAvK@ri#&>peslF?Q>k|8Z8s`+F>0}(7NRL)H6Pm5|y4pPm<nC'
    'Wo9j(sNSy|6q;*3iT~A>{pqWH{l(P-+Bz%#e^~rytc`v!&-'
    'Ll8QSDrB3H%t|YoBeb56SDh|Gz5YC>CXF4Re=r!j4wi52nFordrvXEP*>hgbMh?;6pvwxAZpeaq?Q!woJks8VGlED&FNe%|{a3l('
    'ZC@=ER+09tAeDiiq&FV|8OR&SuzA`MJ*&JGSuhhqK2#a2saXjd~VO`CNzc)y!Z*N9C;%Wf~Hqo_^;{2P<#VRap%Sx>ZgT2-'
    '_NYTy%#dS*=lY*>F)?+L5&fY95qQySH?pEA15G781|k+Ff4zl21qij4nHsrdt)pGWy21sHuJ3^`cDkq&eD`%be6$u^pkOdZ`8)vO'
    '{O3?hK`C?r0gCrOESxJI5Z}%nUS$r=?7tByrg|rm<UlYi|baRN!$XGP$X!Gh8^56?SoV8f;MC@_f7VvZ~|%r~x|+Z(5trx8SRej_'
    'cj{)fK=JK4B~J^1(HOIslCnRrU92Th5BB$8)@HSNi%(N&7OSTmM=?E}r$(g=#58Dn4cdygm)Q)dk%eZyimY5niyZxiOWEd9iFpUA'
    'G9oMEcsb4N-h1+IDpavG455YvK@&fYf3MZAX;><A<t$NPjej$uz4;$H#QUu`-'
    'iJQW_i5_}I~mE%&GxNf=ZH?6ErA@)SmT`WxE|+^5}h6EmX7GDpc((cmc*_h<)hZj9InHEeV4=xp}$&89^<RM0^YW0uHRjlQdw{^m'
    '3>QtlUeuKSzOVS6`SmF>Nw+HK>;8l-'
    'LH4m~x^Y2x{&y5>epXd1uitZ%xvpvyMiLg*(z`?uF}0Uzq%f(1Vws@iHj;;|pCw{{jkSY_5n9R9h3RsDumX{J=4c7byZlWX_GEZY'
    '3PKCp>a_c==J$9g>{EAgw`dv<i&Yo;qwAOB48!DT<Spq{LEGk3t4zL>UF@mJf{>gGapXqw+L2pe4hury)h<p$o_VapfAmKuaG07{'
    '1L=@wY2mwvQ`-jOz0J@>En-d`-2;ByVoW9SztC$x{$vkcH1LY7n<sj@ECFutcdQ=`hFCoziROdb6zOw-AG>I;n>*<2hN!KL`=EVD'
    'x?`O<1EJ2McsvJp>L$1laXsqmxLBdW%thMr-'
    'z;QVPvWS)!7keZT?%Dvi(Ut2jwHkOU|Nug>e^BU05Jf?n3_&#Wtp@&sZmnaH#SQ;Jr(9wX6*Vr2KT-ua36zc}9KUYW5xGe}}8Ray'
    'JuC(K+r#FnEz)b8_CaS>$t^V_O8GTT$U!T-'
    'm8WP%6izTD*7}d6H73=D_AW+%?9X|$(`$iLDcsCyo6Fk%#`I{=zt}D1+*A0j%sqMjnSsUA!wp;9-'
    'agf`ZzW!P@Hb&y{!3y8dw6Owox~+ku4c%&vO3yxcy8||1@;<w8zK3Bb=H4hguFtwD*IlN|cICbupEkG?XRS83BLhHwjWvj2u+-'
    '217~CrZ%S^|YOT$w9n1PO4<wss<zn`QCaWvD(o&{q=mAba7T}zolx-Nb<=H1<cSxX>@70Z>lmu*5KZ|&4En{H^vT=V0t$u-'
    'w7C9h&2dR;xQFx1S5HQzFy4a<w*G3;sWX}JA1o6l*2P4xS}7G6?GVycwU&*LNH42(t_G&MqWiW{|WR%RuN4~@)~awf8Zgf?4t&t('
    'pG>X|!E`}xvPgJwn;Q#GCzY#+u(JYJrdSIG}GCv7!Khk1p>kzsSqS}+cxJ#^tihTY749pzuQ247T~CDV}hktX%AV4fyC^xeuoZ7~'
    '&C|E!DoaCbo_e5}2oa=^_orlur_?GQ(+-a&a(L#_#x4@0ITO}n^$YX{!h0N#|8Y0BUwI>_;9E3Q-'
    '9ykfVU{Z@pfz$>Y8%>=pj1O|FLGUD>nNrWAZjP2J5l(+2{IliZ>l=c&aks8NvXyt|kH``unhPyprqLq(onZZ~Ch|j&@4&0g0n;V)'
    '7$02iL8K|w1j%nEqJhIV<LL0nkfG0^hxwmV#Ls@=))7<yh{aw@U(ew7sK&Jb(rvzT*p~mtc))9jH&r)I9_{_a~s!cBiXYy&7tvon'
    'uRnQAymiAKou4KK>gjh1@vRNe7>~~`6Nq-W-#X7Rc-'
    'u+s4P%Lv)TNp`49qcIBp>Q5+9@0u$WH?{%47Oq9QeigBXh&n}kQuZLz1XBO?}8<<u!KG3t8D4XXQ!*Csx4TceZ=eF7!9j;cOa~>)'
    'VkQJWVN5vPS@7-'
    'H2bxLN@w$kL5hKxeSu?nU8cGzo*Hg0NV2wNviF2D71B2ni5~f+i)ucdjm^uslr53HNZC47d@8>(;`{@yh#$Z4VL6MosY`qWW4rda'
    'qj@5oJ3$VE6g=Ach}<S6*)YCQ#6eMs#<>N<xxvGrWk=q21D@4fjqCd97Hm<1+`QOx7+AI53MnU4=m$0Ls{twx4{hkZ&SFEqSzmvX'
    't#-Ei*8ZM(wCR2$X@bt;tk$$P?_F;b5MEn5c6Qza-'
    'W4{hjk4W~a}Q%SUm8gEGX2)RwB~B2{u^K3L`{}P+)R9<=JZASiP=Fq+W5Mn=kHzOX#l(o2qftZMN-'
    '}FF#YoU?N7#=CU<BKe<$qTQ8M$o7)#z!)<g`x0(dpaZcTA12dGzuma*E+@(0z^edN8MTPLoft7KE#5x?JDwobahb>XqbV;k+Erk>'
    '2>ZEvK`(JT$*`#xottpliZv|^oKc#)9_1ip})rZ(79q%0EEynjEPleK@gh79zTYi^2WFpre6-%+rx-'
    'd3PP<?OBo^2?#&qbzu9CA!cHmyvO0w)Gj&;8trjTKcFG{NcC1y%}qFN+TR8vr`%x7>iDsAI9=1Dxy*CR(&kr*T_u-'
    '?ySruYWQ}?<|ek=BGPOgddEFbziBMSYmJ+=mfo665X%IfHrXkf)JicTW4G11>=RdYEYD6un(18T#~M?&8qOh4I_k@WG^U}IU=>BS'
    'Z^r<1#DHrRRU#<BksoU7xvu+Wf$74l_8wcK%t$O`DZA33!M*yp4w?M3*v)76d+Hd&)ErT6)y}p$2X=$iHA$QrA;sA)&f*GO|NYm0'
    'cs7FOTdGrOyPeqObIMm*-'
    'B`(%O!a*hl$_3rX$3&7p@XeYv4FFA8_PNsQ60xG^cBo(sQGFm+uq#&n=^0uO5H+qwL4&w4;d$G#p1==d83EM?5?hHtfU@Q3&2|uE'
    'cM?@0Kn=a`e~>kn=u))ao5$$VC4Oso>^5@cA<2N2+*3!Y9p7y$r2VI=i0~WG7Uopm9CewfC9)Vw6pBt?7!EU<aUsJwcqd5(Vmc`3'
    'oun=g&(RacWjIy)8OoP8rwCU8|`x?R(kDYy0NKxR6g3{8T*8!9j&mYdMq^VS_TDts~>D;n^rFV8^^=|tee?ygl#ClDFmxufQ=ak<'
    '@?C|U)961xho^9WVWBF8uRpWn~+<&y4eA7mgK|0a?9Kl(~!ft=kk*0GTGbhG!o=)PV6p(z81pL?$U6EUNrM8^Z%}xPUwIM$enn*l'
    'BNpqrIG$;{rC;RL`+$|1PHk*44Sk=HGB6_){d$~);J7g!>;<o_O66iQrnG*(xbQ6jr%9l{5ATnQBo6YZ=Y#zDmkeXIM{o>lPfV}?'
    '|0q1HP(+D=$cF7qjq~H8pxLZhh?n7b=elpov|Z-'
    'V>MKbK+jrcYS@X2>s!XzSgvBrmT4V1XIwcDZ|6Y1e4`q{SQgqC8edxTk$OaCg-M_3T8Fe+8z6I%DRY`noMlHG44Gf;iI>JLF|c+T'
    'mk*-dIA_;%U&H-BG~#9MI2%KfL=6VLZXeQ+EVbTbu$DSjPHi>l1a^r-'
    'ofWpZYT()g;@8Sn<_<8vA|&?tq@;s&T}K*9x7r?WW$B7`P^x`s^=ajsD<zut7@|dWQ?IVqHuXVtSs0p<Y9F@O+ljDde5@Bz>sb4z'
    'GglnzqgV%*`njClW8R>kg^)dxioBke4&GRnDLj^?R=eKU&#7GP-'
    '{2v8@K4|)qvj~)I@NkHtHdCg_x%Yy{dBOUJMWubdu$O?hy13_7=@2DFzA^IrD?l9@6L|sR_e3uE~ve0X|s)?jXCVTy}gm6n$0v)e'
    'r3u#Sv$zfJ(8y3G+YYYsY?8@H5|GlS;jInom~m}tz+H;5nAiGZ3=)ETBv7m*WA&@rR~Zx?6xk)kCq~wt5(dur(G7UBeC>arFZIMz'
    '_NC<54PScEOV%N8%SZW+WB_&WvP8lTW@XS;V$bWW~}|PrEW6k+1K!*S?QsC1Y)HtW_Pl;FAe5w8IlZbv-@~p-><z|4dq)Uk}M~p-'
    'FD<2!_zGNz^QLj6~702N5AS64?`z2vvaSBr8Uj|k(SsFG>B`>1u<U6vu0!kk$v}k8_Tr@?<CigZqfv#dv@E=NlwLOx!e-'
    'jB){sM<q@ofQtVWi!fxFkno<M}wF|Tl+vfmI%J}49#{m0Oux^{<YGbLk_C8xfSVzQlWYl}r{p`62Z&dBuQhQp_uzui-'
    '4OqccU+=zl2}swX4R0njwWH-~<2QKx=IhbgS-}t|^-lQh=zh=u-&$wFynWlG7i*oDzH0M`laAslUyj$9`-'
    'WWWi?Mg$;NR23`fQl^L%$niPV2CD;HfoZ=GBk(u0d_I*U&b5%z3tTYveYeQKN4vVDE{gEeRLQt!Ku}oGjB3X|tfyG}X=UE?$VK&&'
    'Igs?)97dY^i&kyV3~565b}gm(TS+XX^k0TN8@A?w0jMXo|;)RO)gmFDcDO&o3!WfzL0QR(mo|Q)N(+-'
    'tqJH0(O5;V`s`zZ)OHr@2fjT4Fx4T-M3ylonYk5S!?g2jR`%Od!3-k%JDWjn<&>uXO_V5M40AU|MB-'
    '<|J}ox<H9M4IKOm3pXX;PZ#QhC<6~w8tUGIDhZh-fH?J-'
    'FVlP;xYm=7`>e}U*>&=@gMzTNs;u;y&mZ=_Zdw1m%d$w3C%Or~ind@A+H*dT3L%S3Qi^rS3{qXLm@816S(}#~AKK}CK`?v2tzWw('
    'PKm8{@<=mS&F3arW5@QO{r`OY)B|iP(@b7f`H^{Fh`NQQue13g?<K4P0arI$MtDo{4rqi#cfN%6+UKT$GKc9c&-'
    '{6&qk9qIsZCRfm^PA~SoaT9(WG%}Y79ZrbWm(tBCm-FshIw=H+7veV+Bwb<kC@^V<<)g{A*N;7+_vJ`vWeud#c9Ff;<C>3B5%Xfe'
    'Oet>5Msdc^A{!;@s7=J+ZN_|_OIQW2^(6|5*;44P7d#LZ)7i6bDUT2d<v56^!~^1fBx{@`&VDRTDNtIZkliqXL|qq_S;|He)sMdI'
    'i(d}nU-l<@!T-'
    'YX*q2@uXH;6FP0ZJmt;>M!m_wUUc$Z{_8YK)6sJtl<dR!KnGNT%y1a@dZOKi+xj8w%2E4SoX`3hS68=B?b<IZsy5_ffC*Lo}J*SY'
    'ujL*e&*{03QNAcsbrj){l&%6D^+!W^o>R=H#jpW6Fo$<~YtJ`p}F*$ch4u7*f7oIq9@+UsRd3ra`AZ3(ATId5_$KfXI9b3Z*WCDS'
    'KC%5@!j+<~~rV!S?q<O+ivj<T#&(8rg5&m-'
    '6mMj7v!=9ED!TL@(S$Gg^3wBA<4E~yCA?^&mTtL@p+wcYRoGCQpvoj7MYyqpu6pFz`@W-'
    '}#5s>U3$2ZeFL54PL6ttb=6^RTqS|+e0w(78E?<je|@EaE7an>ox>93h$4pIUdLrp*)Vt&n~#V-'
    '!a3Q!zV=8UxB48yWQgh02<Q!Y42B6JJZfW>FMvMo#8mI&e}*<mIaP9r5eA>qwylDFXyHX)$M=d@)pQCSK^Ok04&u8>h5V<wp2)>+'
    '6cav&#636=_iK@s9Bp)_zT@mWM`BumpK#{(kG0g^wTm`oJd1Q~;L#H=?QRM>*m;v_Z~mYvD4KKvY@%%B&hO%^2Nf&k)=02=sAvxH'
    'wldax%DVOMmE$a&nruh4Tz&NWQS48`J8n&iOJE{7{<zXbf1|IW&9N?-)%<-'
    '_ZQa{$rO8)&i(G8pj!7S~O*uwn~YW}vF4DKtbeZL)8WM7Ds%?tJDT3s5hco7_n*rwO!xSpoyckeLOl2WlKA3(=bW=JRqqR2;VGwh'
    'hvlWdM8#g5kX|0<Z;{+UpjD%Y<XF+8|OGpu(|jCufzwzPrKjLS@*qO>E*$Ggt(uHa~3>_7{968NLa}0EwCMb}mWgt_TpGwAls8W@'
    'u_`Yk}4apxZk65zz!Qt)k8*F>s=OglM22gmr;zV#Rsqpam2V2sTepK}%L<ct^mVq2WUSAM8Ynux((eAhh$laIUcII00}ewwfJ;-'
    '~t32)SiW<SD_<L06GrFBY@R*YD1fX{?0))2GL+K(+t*oa|?`xeE)gRJI%aGv-Rn}24qvCSpb(n!v>IST19fQY(mXD$OWV@ZZS(3b'
    'Ud{GES?eAXv#E$UQV0Hm=ht%QUWg}!TAB}z*x8w$*@9bGRQPY7c<GA&O-'
    'ukgTDA9dsoTl)y;S~bSXGlh_*uT;T_>mxHmCKc(A;|X-'
    '+9~9?adg^b)P+ZRs?FN%9a5oE+o^GM)Jha+Aadz$U>mTc#Nlkb~`mZvf8P-'
    'LD09CJ9CIrox(Owoahb1ceCOwrrVXVhrGRLDeMqQM+t{A3=L4O!yF!Yz-m@KW4)LfZ-rvnPgZN7;J{pz+$oukVk>4at0}4k(tewO'
    '&08+hoOz0c6{qpWZGrr=AOyr@LBi+u*nb}(N9C<f)$*^T;X6>JRy?{TbQC3ofiGPQ)_{R^IL)oCjNn}W)c$*8v#Oy^RP=LbQmNq;'
    'rLS8J>pe(2A}leNWvM-<btG%=7L88Q#|eTf?|r`F%T9koLLMU4HW^2fEL7g<!gupw-'
    'hc4NQno*FvJUrU8%C{+?`2w4gx@V@tvRzAX`4g0h0;BAe4YPgi3_o-Sz(rdjg&Au^&k6ZZff9Sa-'
    'b3i}Dc`^CB=Cj4CuZ9EmJkcsJMrp$)c=aA229C@ILmGQ&Rr{|GCJ`Hf-'
    '2v4CxHTv<b>4KKkagWKWC?A{GXgXQ6iV5Q_(v!el?T_@3TE0`s@8qwxwqWLSTK?#bB2%QP<CCv@0!A^>0gq@9X#!hysVPWzMsMs~'
    '01QTx)SzjTY&^m7SW+6ui0KhNU#hN8yTj1?OgF_>W&KIIVXN&cPR|UUm-8l`a-'
    '@!g*X#|7G24GvD*K**6L(7btB!GmqfomT`!DX7k!-'
    'Ge+ZP5M@&6sr=js=kjR4RNBxZ<b%?cwI710T+M?3!p`aV!Xu0DooLt~t$vMgelLiF1@Bz&gDQZ!?Nxf!Ge`nkuhmj`LzJ@B&mm1V'
    'n!g+ZCS#HWiWzKFlT%MpLXX*ia}lW!nUq6Iy|7uwLlOouJ?@s5v?J<g!|W*vt@Uya}ol$0+n%h1&S=0!I>xQa*|En`P?~V`1&0>D'
    'D(;n;XIitOefo+1XtR+YrN<lo#=Pz#(nowc{+YJMVPStx>S>`arG;Zqp1i19cUaUAD1iaoE682q+ON?i7To$iW~yi`^<gGABp`{3'
    '407);&rBr#La4AjCH9@f9|jqa%qv=B#YQYk??1o+O5kYi3Aj2yh<$@e-'
    '700><p^AP!sOoMm`>6S4$|a|BtBm*m@x=##^mH1ZMnqoMJ3`oSJTvBSPe%m>Asi6ssk0_jEMDeYnop@hPM+W~2UgPu1N#TZPY%Ao'
    'uf;Wm&P?}0apx5jvi!;{3BAmEb3P0nT~h%dMbAd(QXW~;NzIfUNC+s?5+j0^r*<q@*2{K0@1JlrRV5a3GAI|2OynR8xrp~N>;3TA'
    'J~tpx+Zix%$(K{M7J=RN|8Lj&!NU=)N{rc6LASG+EW4SW-Sl503#8&(N@84*$LmIR!Nhpt(nx-'
    '1`ven4JG7`_f%Ca>TJG0M<y5_|4;8|wsHEfU2dzRyt`ehC|%#8<}eVitZ6K66{lIssyn4-'
    'GO6j)SWR?FFYhQ&89&d&fq$h?uuZ!DrfC5l|>2?kBX=MSQQ2Ofez4a4VlJaD+@~aa`bVK(d2BDNB5NgfoUW!+UW%qhi1C6(M5qZS'
    'v6xc!sb^qEK;bGwI>+Vr>#*!M{c*{F_X{O&C`cy_g!=H4YpH8zlDIa$qN(I>HXPdpJEr8}kluO@jNupNM!Y4i|)W*sPP74iXQK(F'
    'cD;>2yQsQdK%U7WNteqXd#mi#t1LTZA>(0c-{=3djNt<4!=;B+@TF8g!L-mYKZpm}hxu5?3h$RZ;vKsAMQl_%`#-'
    '+2HC3bH8~ocG~t(`X&e$`NVNRuVz2g=Rgs%1y^Vh&j~!XukH4eJ9WVrU=W-PJ_w~i`M9Jm>|yqzHZgE1U62rGj?)sHdQQR-'
    'O|w*dK*$*de}jQp#O-'
    'n0PEJtK2>|1=Nyy~H|JDiX7tA;1;0<mRm<UV{#VKCCkW&m!fG;UH@U+8NN!%w;8|Ev*)5;EV5T1ld6HH9bEYlMfUCg-PRwr(Sihz'
    'yEU;@Nq+je?N__;`Qo>j(`Wxv%5Yhjr1O%mivOa$u~BBQ4gC|AO5|3>07IQxh_AZubPv?v?oh8!b;7B<{q;xhsUkz8jv0nw9g-'
    'pM^fz2Z$c8`1ySa^s6ZLJ${@KB$+uHwnIhm<nt@?`AAXEDFaJ%Mtt}YhfR<*93C{4Fr=Z^nztWbRGoq61A5-'
    '2!z9s128cb?EQ({VafO`Gz~1AU$`1B3BJfZ1N=U?QR0k~+m0%^Z@F319%U2&Vq@b7-'
    'Go@~32u~769xunglB+>P6iw&3Ti<g&wJFE1ltye2)YPDkckabEU=rnoA6I|ffpwtO=9>Kv84IJ1Zp9H!$p;You+&QI5T*pI7?_6n'
    '7uuOfI!1W$c@beA=#-7Iw5-'
    '91P8)%R;r3MYJ+|dE%=X{sHJd0?t*En^wxlp($ERNz&{tfGznwGT@~Y~g&6Q%Rza&XeKW)}!?J*tT((U~E_npffkOpF>GDh9KMA?'
    ';pYXzVF~Hwj6sH+sfAA*&Z3RySaX!G184?kn7KDQYAdEU0$=k*4<Q2;YUWC60m9$>_wZ2W>YCvC*$IQ&-'
    'eW1?F;6&IexpP2_2#LxJ4(&a`5J6_47Pj4)&kz8DDrf*Wy;=EV(Ga~P9tZ;JJr01g3t<;@SVGoi_re@(;kF5f3Vx})Boqt_mM<D%'
    '+FeVm0}L!*EVQ1cbtg4!a>T*IHJ;>Du1H<@zvTrg)CTw!kryZ>Hm6SNx<*mr5Y0E1;8QMfW`Ppn;Le%5#qxOJMmz^+y@NzBS2z#s'
    'R&dCC=>)Yf!^;$ZVg?I`tURF0*92SX;Rx=?6lg5Bjfl=fKl>9zC(3+>Kb@ck`BDS?GKuSmlg^4l9LgwK3z{99Moc606|(^*5jGa!'
    'cx%u%l<u+I4m&Tm1|eQDvSP`M{;`e|W``Bq7VCl|o!1>e-'
    'z4C~wuN><P9_rsQm|$hV#>jVL~l@r;z8lb8@#8CEl*311!oD__Si!(0eFTHsu|XBzcwtMno%1(bdi7uPfa^@CB%}*LjcP)?8J3~*'
    'eqC<+_i5DCvF-3in9kJ1Dc62hKtL8SPv&LAtEEWor!hoDozG01~^NWhQon4>(mxrgeHk{FA?`xV-q7R=0~DrXxaVl2FH&-'
    '1U;8~!hC~~{36$6UO02x_9!BX4TTs+iOP4x1(e?;XvrpY$4}up1Tl-*FWdHua*3nYBp1;L7q?R&1Z}WNumQ&$#&{>U@R8h-'
    '!#c!iu2DO!PXO!DREt@lgGum!87u;SM|nlp6bPU2HgE+t7N+%NY6QxcyASct*0|qUXFCpRO!@nkT{lA`MhSl>35s(pAaJ~R!aiS4'
    '=FiUJu89``%?CA}$snPQSiVUD^|b47sP9Fvd`Z}lXmiJu@J2B`aw!E@cc(A3go8ekyJaZOR3mV#B(o&ByaJm_7Q{BEnNYaO&bm`k'
    'B0Z8)<K;5k3q?It=6+R|V`_XSXywCBchQCddP|Z7mH-'
    '|X3aoN#38xvBQ5ZY>x^RVX2ncw+xFwnU9io^y2p?vUCevMXfFOQ?0mE!%Ctd_jAdMhC@wP)zph1GV58~7=;iQkTz*!<Sh(5UG%C$'
    '(BuEib!-YVpJ(gsW+bQ-'
    ')BoUBL~j^3=1%kwFpBNT^wngkh=KzZ|iM{{6l@Ld^yh}mh&%43P5kKh!`Md$8>Ecvb$G=o~mVv46rCWs^qxgC|0AQ3Btvnw9Oayn'
    '=jG&wy9rQyotfJ?BpNiOFF^4+m~Xo(2l2SOns<4Hlw4Rx5{Cr4PY<w`-'
    'Pt4oH<yLsn(J&4NHs^)6g^oPI)l3XBu*OF2?87fJ2k+dtg{kydZf&)pC$eP10P#RVR!8gJ+9MmdT7Fl`7r3T`gO#q(VZvneX_Ltn'
    'qidEQ8*g*@!L560=;X~I558-H?+yyz&&naZ=N)8BXm?iH=0`C*Cg@FWF-sCK{OJaxh_8RP&+BNnizDAYa3TGGeR+R~uOfnttaKs#'
    '^w1o`Qg8y+alCqX%6*@dao{~o)Ysk7JCqZC6a)&PwN4AGaQrxhYxPgE>$t5#xiQ;!hxl7u$IMj6#3y>t#cR3`GT?l~@X(FD^>@9X'
    'y93e0i0>)Lc7eBs#_rt5NUj6XlKfeF?>aTx&{p#P}zx()?pWeUy`Qy7kfA{XE@4o-'
    'P?>>J2;m@zW`ma|%ynlzk|IZ4)9QG7q36&(ER`&WKHX?OmboMkO*nhbw5sX`QlrfWeyZ+y$s+uLsZ4tL&6(-'
    '40SdhDRNz(A*FU9>93M5-'
    'VT%mgo<^)f}5v*eQAn)fG5?kCPo|bcr)201^Ga<q(_dEhV<n*uzvnL0PSlEkwd<xFMFOy{Oh=^=i=PZjT<4sbFgs|Iwd*o8SM}fl'
    ';_8cFmGH4JXI5-'
    'PL$^IWC8S#6RXb8u?KHrGANy0stov>eIdDzdeiI19gMHl7fbiq5}@w+{PPz*H8uV~3xlwKAFxn@S}1vBF0x>hzTewP4U0)p=zgIt'
    '>WIlFO^tK`54KmXgiKfTB4LMq}UIKw8M72JhA{YY+I7f}+>CplR4LfQnH3mA(nWFrIrTAT+MZs=!x|7-'
    'XDH?Z;hpW<`Nof<es5WxL7C&^pfB+qmfUA6883)WHmy(mn)?)i2B+??|ueGu5;{MABxkX?dANstt*;p&+<0Qe%%h?@lLaW~OXvck'
    'ORc)2)y`X+eGV!xot)+os+^5xJCa`y^yO7ho!5HaUI&2q!GXAHn9;G85J5Kw$i3Yp~s19n`j=Cou-QxaAtvDk2$SFuc4r-'
    '5UatZ@VZ-9IY+6By2|lDd<R--)&ny9K8Y-j~l99W*ZdFVIXJrv0i;ZkFL@$X&@Ug+cN$oG3Zj$`Zf%>r=+^+i$;rm-AwOhQy0_Px'
    '8liKYai3r(fRw32*uE>Z_a%vxuuJkf|uw?e(ia{pH`jfB)yV-@SkP(_g-'
    'Q|HF?zz0Yri1bFz_nTJKs;W2;y%ez0l{qaxlev!QG;;UA9Pu4qM6`yukrYqMYGFf}R__{tyJS_K`uz=+Prt<?!$V7A>blxv-'
    'zy0v%zx*ukj(`1s#KuxZ'
)
_V18_RUNTIME = json.loads(
    zlib.decompress(base64.b85decode(_V18_RUNTIME_B85)).decode()
)
# END GENERATED V18 RUNTIME

def _spread_animals(cows, sheep):
    total = min(len(ANIMAL_SITES), max(0, int(cows)) + max(0, int(sheep)))
    sheep = min(max(0, int(sheep)), total)
    plan = {}
    sheep_used = 0
    for i, pos in enumerate(ANIMAL_SITES[:total]):
        target_sheep = round((i + 1) * sheep / total) if total else 0
        animal = "SHEEP" if target_sheep > sheep_used else "COW"
        sheep_used += animal == "SHEEP"
        plan[pos] = animal
    return plan


def _build_animal_plan(cows, sheep):
    """Build a target herd while allowing a distinct four-animal opening."""
    cows = max(0, int(cows))
    sheep = max(0, int(sheep))
    total = min(len(ANIMAL_SITES), cows + sheep)
    opening_cows = STRATEGY.get("opening_cows")
    opening_sheep = STRATEGY.get("opening_sheep")
    if opening_cows is None or opening_sheep is None:
        return _spread_animals(cows, sheep)
    opening_total = min(4, total, max(0, int(opening_cows)) + max(0, int(opening_sheep)))
    opening_sheep = min(max(0, int(opening_sheep)), opening_total, sheep)
    opening_cows = min(max(0, int(opening_cows)), opening_total - opening_sheep, cows)
    opening_total = opening_cows + opening_sheep
    plan = dict(list(_spread_animals(opening_cows, opening_sheep).items())[:opening_total])
    remaining_total = total - opening_total
    remaining_sheep = min(max(0, sheep - opening_sheep), remaining_total)
    sheep_used = 0
    for i, pos in enumerate(ANIMAL_SITES[opening_total:opening_total + remaining_total]):
        target_sheep = round((i + 1) * remaining_sheep / remaining_total) if remaining_total else 0
        animal = "SHEEP" if target_sheep > sheep_used else "COW"
        sheep_used += animal == "SHEEP"
        plan[pos] = animal
    return plan


def _build_opening_plan(wheat, melons, animal_plan, carrots=2):
    """Build a 21-tile NW opening with long crops nearest the shed."""
    blocked = set(list(animal_plan)[:4])
    slots = [(x, y) for y in range(5) for x in range(5) if (x, y) not in blocked]
    slots.sort(key=lambda p: (abs(p[0] - 4) + abs(p[1] - 4), p[1], p[0]))
    melons = min(max(0, int(melons)), len(slots))
    plan = {pos: "MELON" for pos in slots[:melons]}
    remaining = slots[melons:]
    carrots = min(max(0, int(carrots)), max(0, len(remaining) - int(wheat)))
    for pos in remaining[:carrots]:
        plan[pos] = "CARROT"
    for pos in remaining[carrots:carrots + max(0, int(wheat))]:
        plan[pos] = "WHEAT"
    return plan


def _build_crop_plan(strawberries, animal_plan, tomatoes=0):
    # Melons retain their proven two-cycle opening sites.  Every other usable
    # tile becomes a candidate strawberry site, prioritized near the shed.
    plan = {pos: crop for pos, crop in OPENING_CROP_PLAN.items() if crop == "MELON"}
    opening_strawberries = [pos for pos, crop in OPENING_CROP_PLAN.items() if crop == "STRAWBERRY"]
    candidates = [
        (x, y)
        for y in range(10)
        for x in range(10)
        if ((x < 5 and y < 5) or (x >= 5 and y < 5) or (x < 5 and y >= 5))
        and (x, y) not in animal_plan
        and (x, y) not in plan
    ]
    candidates.sort(
        key=lambda p: (
            0 if p in opening_strawberries else 1,
            abs(p[0] - 4.5) + abs(p[1] - 4.5),
            p[1],
            p[0],
        )
    )
    for pos in candidates[:max(0, int(strawberries))]:
        plan[pos] = "STRAWBERRY"
    tomato_start = max(0, int(strawberries))
    for pos in candidates[tomato_start:tomato_start + max(0, int(tomatoes))]:
        plan[pos] = "TOMATO"
    return plan


def configure_strategy(overrides=None):
    """Configure one module instance for local HPO; submission uses defaults."""
    global STRATEGY, ANIMAL_PLAN, CROP_PLAN, FIELD_PLAN, OPENING_CROP_PLAN
    global ADAPTIVE_ANIMAL_PLANS, ADAPTIVE_CROP_PLANS, EXPERT_PROFILES
    global _OPPONENT_STYLE, _EXPERT_EVIDENCE, _MARKET_ANIMAL_SHARE, _PLAN_CACHE
    global _V11_SELECTED_RADIANT_VARIANT
    global _V13_MARKET_MODE, _V13_MARKET_CONFIDENCE, _V13_MARKET_LOCK_UNTIL
    global _V14_MARKET_MODE, _V14_MARKET_CONFIDENCE, _V14_MARKET_LOCK_UNTIL
    global _V15_MARKET_MODE, _V15_MARKET_CONFIDENCE, _V15_MARKET_LOCK_UNTIL
    global _V16_MARKET_MODE, _V16_MARKET_CONFIDENCE, _V16_MARKET_LOCK_UNTIL
    global _V18_SELECTED_MARKET, _V18_SELECTED_DAY, _V18_SELECTED_BOARD
    STRATEGY = dict(DEFAULT_STRATEGY)
    if overrides:
        STRATEGY.update(overrides)
    ANIMAL_PLAN = _build_animal_plan(STRATEGY["cows"], STRATEGY["sheep"])
    OPENING_CROP_PLAN = _build_opening_plan(
        STRATEGY["opening_wheat"], STRATEGY["opening_melons"], ANIMAL_PLAN,
        STRATEGY["opening_carrots"],
    )
    CROP_PLAN = _build_crop_plan(STRATEGY["strawberries"], ANIMAL_PLAN)
    livestock_plan = _build_animal_plan(STRATEGY["livestock_cows"], STRATEGY["livestock_sheep"])
    premium_plan = _build_animal_plan(STRATEGY["premium_cows"], STRATEGY["premium_sheep"])
    ADAPTIVE_ANIMAL_PLANS = {
        None: ANIMAL_PLAN,
        "WHEAT_RUSH": ANIMAL_PLAN,
        "LIVESTOCK_RUSH": livestock_plan,
        "PREMIUM_CROP": premium_plan,
    }
    ADAPTIVE_CROP_PLANS = {
        None: CROP_PLAN,
        "WHEAT_RUSH": CROP_PLAN,
        "LIVESTOCK_RUSH": _build_crop_plan(
            STRATEGY["livestock_strawberries"], livestock_plan, STRATEGY["livestock_tomatoes"]
        ),
        "PREMIUM_CROP": _build_crop_plan(
            STRATEGY["premium_strawberries"], premium_plan, STRATEGY["premium_tomatoes"]
        ),
    }
    base_profile = {
        "hands": STRATEGY["hands"],
        "cows": STRATEGY["cows"],
        "sheep": STRATEGY["sheep"],
        "strawberries": STRATEGY["strawberries"],
        "tomatoes": 0,
        "cash_reserve": STRATEGY["cash_reserve"],
        "animal_cap": STRATEGY["animal_daily_cap"],
        "strawberry_last_plant": STRATEGY["strawberry_last_plant"],
    }
    EXPERT_PROFILES = {
        "BASE": base_profile,
        "WHEAT_RUSH": dict(
            base_profile,
            cash_reserve=STRATEGY["wheat_rush_cash_reserve"],
            animal_cap=STRATEGY["wheat_rush_animal_cap"],
        ),
        "COW_RUSH": dict(
            base_profile,
            cows=STRATEGY["cow_expert_cows"],
            sheep=STRATEGY["cow_expert_sheep"],
            cash_reserve=STRATEGY["livestock_cash_reserve"],
            animal_cap=STRATEGY["livestock_animal_cap"],
        ),
        "SHEEP_RUSH": dict(
            base_profile,
            cows=STRATEGY["sheep_expert_cows"],
            sheep=STRATEGY["sheep_expert_sheep"],
            cash_reserve=STRATEGY["livestock_cash_reserve"],
            animal_cap=STRATEGY["livestock_animal_cap"],
        ),
        "PREMIUM_CROP": dict(
            base_profile,
            cows=STRATEGY["premium_cows"],
            sheep=STRATEGY["premium_sheep"],
            strawberries=STRATEGY["premium_strawberries"],
            tomatoes=STRATEGY["premium_tomatoes"],
            cash_reserve=STRATEGY["premium_cash_reserve"],
            animal_cap=STRATEGY["premium_animal_cap"],
        ),
    }
    FIELD_PLAN = {pos: crop for pos, crop in OPENING_CROP_PLAN.items()}
    _OPPONENT_STYLE = None
    _EXPERT_EVIDENCE = {}
    _MARKET_ANIMAL_SHARE = None
    _V11_SELECTED_RADIANT_VARIANT = None
    _V13_MARKET_MODE = "BASE"
    _V13_MARKET_CONFIDENCE = 0.0
    _V13_MARKET_LOCK_UNTIL = -1
    _V14_MARKET_MODE = "BASE"
    _V14_MARKET_CONFIDENCE = 0.0
    _V14_MARKET_LOCK_UNTIL = -1
    _V15_MARKET_MODE = "BASE"
    _V15_MARKET_CONFIDENCE = 0.0
    _V15_MARKET_LOCK_UNTIL = -1
    _V16_MARKET_MODE = "BASE"
    _V16_MARKET_CONFIDENCE = 0.0
    _V16_MARKET_LOCK_UNTIL = -1
    _V18_SELECTED_MARKET = {0: None, 1: None}
    _V18_SELECTED_DAY = {0: None, 1: None}
    _V18_SELECTED_BOARD = {0: None, 1: None}
    _PLAN_CACHE = {}


def _expert_weights():
    forced = STRATEGY.get("force_expert")
    if forced in EXPERT_PROFILES:
        return {forced: 1.0}
    # Wheat-capital evidence represents an existential feed-price risk and
    # therefore owns the portfolio once confirmed.  Other regimes blend.
    if float(_EXPERT_EVIDENCE.get("WHEAT_RUSH", 0)) >= 0.8:
        return {"WHEAT_RUSH": 1.0}
    # Pure early sheep openings depress wool economics before a later herd is
    # visible.  Replay counterfactuals consistently favored keeping the base
    # 8/6 portfolio with extra liquidity, so this signal must act early.
    if float(_EXPERT_EVIDENCE.get("EARLY_SHEEP", 0)) >= 0.8:
        return {"PREMIUM_CROP": 1.0}
    evidence = {
        name: max(0.0, min(1.0, float(_EXPERT_EVIDENCE.get(name, 0))))
        for name in ("COW_RUSH", "SHEEP_RUSH", "PREMIUM_CROP")
    }
    # A farm that exposes both livestock regimes is rotating its market
    # pressure rather than specializing.  Chasing both sides produced the
    # weakest possible 7/7 herd in replay validation; the liquid balanced
    # expert is the robust response to this phase-changing portfolio.
    rotation_threshold = float(STRATEGY.get("rotation_evidence_threshold", 0.9))
    if evidence["COW_RUSH"] >= rotation_threshold and evidence["SHEEP_RUSH"] >= rotation_threshold:
        return {"PREMIUM_CROP": 1.0}
    active = sum(evidence.values())
    if active <= 0:
        return {"BASE": 1.0}
    # A clear signature selects the validated counter exactly.  Lower
    # confidence produces a genuine mixture with BASE, avoiding a brittle
    # all-or-nothing switch on borderline farms.
    expert_mass = 1.0 if max(evidence.values()) >= 0.9 else min(0.85, active)
    weights = {name: expert_mass * value / active for name, value in evidence.items() if value > 0}
    weights["BASE"] = 1.0 - expert_mass
    return weights


def _blended_targets():
    weights = _expert_weights()
    numeric = {}
    for key in ("hands", "cows", "sheep", "strawberries", "tomatoes", "cash_reserve", "animal_cap", "strawberry_last_plant"):
        numeric[key] = sum(weight * float(EXPERT_PROFILES[name][key]) for name, weight in weights.items())
    total_animals = max(0, round(numeric["cows"] + numeric["sheep"]))
    cows = min(total_animals, max(0, round(numeric["cows"])))
    if STRATEGY.get("price_adaptive_animals") and _MARKET_ANIMAL_SHARE is not None:
        cows = min(total_animals, max(0, round(total_animals * _MARKET_ANIMAL_SHARE)))
    return {
        "hands": max(0, round(numeric["hands"])),
        "cows": cows,
        "sheep": total_animals - cows,
        "strawberries": max(0, round(numeric["strawberries"])),
        "tomatoes": max(0, round(numeric["tomatoes"])),
        "cash_reserve": max(0, round(numeric["cash_reserve"])),
        "animal_cap": max(0, round(numeric["animal_cap"])),
        "strawberry_last_plant": max(0, round(numeric["strawberry_last_plant"])),
    }


def _crop_plan(day):
    if day < int(STRATEGY.get("crop_transition_day", 5)):
        return OPENING_CROP_PLAN
    targets = _blended_targets()
    strawberries = targets["strawberries"]
    if STRATEGY.get("strawberry_staging"):
        stages = ((3, 3), (4, 5), (5, 7), (7, 9), (9, 15), (10, 18), (11, 32), (12, 44))
        staged = strawberries
        for final_day, count in stages:
            if day <= final_day:
                staged = count
                break
        strawberries = min(strawberries, staged)
    animal_plan = _build_animal_plan(targets["cows"], targets["sheep"])
    key = (targets["cows"], targets["sheep"], strawberries, targets["tomatoes"])
    if key not in _PLAN_CACHE:
        _PLAN_CACHE[key] = _build_crop_plan(strawberries, animal_plan, targets["tomatoes"])
    return _PLAN_CACHE[key]


def _animal_plan():
    targets = _blended_targets()
    return _build_animal_plan(targets["cows"], targets["sheep"])


def _style_setting(base):
    """Return the soft-gated strategic target for a shared executor."""
    targets = _blended_targets()
    if base in targets:
        return targets[base]
    return STRATEGY[base]


configure_strategy()


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _copy_action(action):
    """Copy a scheduled action before an observation-dependent overlay."""
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }


def _farm_pipeline(farm):
    """Estimate near-future market exposure from public farm assets.

    Opponent inventories are private, so this is deliberately a portfolio
    estimate rather than a prediction of the next exact action.  Yield already
    waiting on a tile receives extra weight, while recurring assets retain a
    smaller baseline weight even between production days.
    """
    exposure = {
        "WHEAT": 0.0,
        "CARROT": 0.0,
        "TOMATO": 0.0,
        "STRAWBERRY": 0.0,
        "MELON": 0.0,
        "EGG": 0.0,
        "MILK": 0.0,
        "WOOL": 0.0,
    }
    animals = 0
    unfed = 0
    for row in (_get(farm, "tiles", []) or []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            ready = max(0.0, float(tile.get("yield_units", 0) or 0))
            crop = tile.get("crop")
            if tile.get("kind") == "PLANT" and crop in exposure:
                # A live crop represents future supply; ready produce is much
                # more likely to collide with our sale in the next few turns.
                exposure[crop] += 1.0 + 2.0 * ready
            animal = tile.get("animal")
            product = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}.get(animal)
            if product:
                animals += 1
                unfed += int(not bool(tile.get("fed_today", False)))
                cadence = {"EGG": 1.0, "MILK": 0.5, "WOOL": 1.0 / 3.0}[product]
                exposure[product] += cadence + 2.0 * ready
    # Feed demand is the only visible buy-side pressure worth considering.
    # Unfed animals increase urgency but do not prove that the shed is empty.
    exposure["WHEAT"] += animals + 0.5 * unfed
    exposure["ANIMALS"] = float(animals)
    exposure["UNFED"] = float(unfed)
    return exposure


def _opponent_pipeline(obs):
    farms = _get(obs, "farms", []) or []
    player = int(_get(obs, "player", 0))
    if len(farms) != 2 or player not in (0, 1):
        return {}
    return _farm_pipeline(farms[1 - player])


def _interference_value(obs, product, quantity=1, pipeline=None):
    """Relative denial value used only to sequence existing v8 sales."""
    pipeline = pipeline if pipeline is not None else _opponent_pipeline(obs)
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    exposure = max(0.0, float(pipeline.get(product, 0.0)))
    price = max(1.0, float(prices.get(product, 1.0)))
    quantity = max(1.0, float(quantity or 1))
    # Exposure captures the opponent's likely supply; price captures how much
    # acceleration is denied if our sale reaches the shared market first.
    return exposure * price * min(quantity, 10.0)


def _safe_wheat_squeeze(obs, market_orders, pipeline):
    """Optionally add one strictly gated feed-denial order.

    This is disabled in the submitted defaults until it beats the unchanged
    v8 schedule out of sample.  Keeping the gate here makes that hypothesis
    directly testable without weakening the production executor.
    """
    if not STRATEGY.get("interference_wheat_squeeze") or len(market_orders) >= MAX_ORDERS:
        return market_orders
    day = int(_get(obs, "day", 0))
    hour = int(_get(obs, "hour", 0))
    if not (8 <= day <= 24 and hour == 0):
        return market_orders
    farms = _get(obs, "farms", []) or []
    player = int(_get(obs, "player", 0))
    if len(farms) != 2 or player not in (0, 1):
        return market_orders
    own = farms[player]
    opponent = farms[1 - player]
    if float(_get(own, "money", 0)) < float(STRATEGY.get("interference_wheat_min_cash", 10000)):
        return market_orders
    if float(pipeline.get("ANIMALS", 0)) < float(STRATEGY.get("interference_wheat_min_opponent_animals", 10)):
        return market_orders
    # Attack only a genuinely liquidity-sensitive herd, never a rich rival.
    if float(_get(opponent, "money", 0)) > 250:
        return market_orders
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    if float(prices.get("WHEAT", 10 ** 9)) > float(STRATEGY.get("interference_wheat_price_cap", 30)):
        return market_orders
    private = _get(obs, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    own_pipeline = _farm_pipeline(own)
    own_animals = int(own_pipeline.get("ANIMALS", 0))
    if int(shed.get("WHEAT", 0) or 0) < 2 * own_animals:
        return market_orders
    if any(order and order[0] == "SELL" and len(order) > 1 and order[1] == "WHEAT" for order in market_orders):
        return market_orders
    units = max(0, min(1, int(STRATEGY.get("interference_wheat_units", 1))))
    if units:
        market_orders.append(["BUY_PRODUCT", "WHEAT", units])
    return market_orders


def _apply_market_interference(obs, action):
    """Apply a market-only overlay without changing farm execution."""
    copied = _copy_action(action)
    if not STRATEGY.get("market_interference"):
        return copied
    pipeline = _opponent_pipeline(obs)
    if not pipeline:
        return copied
    orders = copied["market"]
    if STRATEGY.get("interference_sell_first"):
        targeted = bool(STRATEGY.get("interference_targeted_sort"))

        def priority(pair):
            index, order = pair
            is_sell = bool(order) and order[0] == "SELL"
            if not is_sell:
                return (1, 0.0, index)
            product = order[1] if len(order) > 1 else ""
            quantity = order[2] if len(order) > 2 else 1
            if STRATEGY.get("interference_preserve_wheat_order") and product == "WHEAT":
                return (1, 0.0, index)
            if (
                STRATEGY.get("interference_collision_only")
                and float(pipeline.get(product, 0.0))
                < float(STRATEGY.get("interference_min_exposure", 0.5))
            ):
                return (1, 0.0, index)
            value = _interference_value(obs, product, quantity, pipeline) if targeted else 0.0
            return (0, -value, index)

        orders = [order for _, order in sorted(enumerate(orders), key=priority)]
    copied["market"] = _safe_wheat_squeeze(obs, orders, pipeline)[:MAX_ORDERS]
    return copied


def _v13_market_mode(obs):
    """Select a market expert from public supply with daily hysteresis.

    Production and movement remain on one coherent route.  Only the ordering
    of already-planned sales may change, so a classification error cannot
    invalidate future farm actions.
    """
    global _V13_MARKET_MODE, _V13_MARKET_CONFIDENCE, _V13_MARKET_LOCK_UNTIL
    step = max(0, int(_get(obs, "step", 0)))
    hour = int(_get(obs, "hour", step % 24))
    if step == 0:
        _V13_MARKET_MODE = "BASE"
        _V13_MARKET_CONFIDENCE = 0.0
        _V13_MARKET_LOCK_UNTIL = -1
    if not STRATEGY.get("v13_market_adaptation", True):
        return "BASE"
    if step < _V13_MARKET_LOCK_UNTIL or (hour != 0 and step > 0):
        return _V13_MARKET_MODE

    pipeline = _opponent_pipeline(obs)
    values = [
        max(0.0, float(pipeline.get(product, 0.0)))
        for product in ("CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
    ]
    total = sum(values)
    concentration = max(values, default=0.0) / total if total > 0 else 0.0
    scale = max(0.1, float(STRATEGY.get("v13_gate_exposure_scale", 6.0)))
    target_concentration = max(
        0.1, float(STRATEGY.get("v13_gate_concentration", 0.50))
    )
    confidence = min(1.0, total / scale) * min(
        1.0, concentration / target_concentration
    )
    _V13_MARKET_CONFIDENCE = confidence
    threshold = max(0.0, min(1.0, float(STRATEGY.get("v13_gate_confidence", 0.70))))
    next_mode = "COLLISION" if confidence >= threshold else "BASE"
    # Require substantially weaker contrary evidence before leaving an active
    # specialist.  Public assets normally change slowly, but this also covers
    # deliberate mid-game strategy reversals.
    if _V13_MARKET_MODE == "COLLISION" and confidence >= threshold * 0.5:
        next_mode = "COLLISION"
    if next_mode != _V13_MARKET_MODE:
        _V13_MARKET_MODE = next_mode
        _V13_MARKET_LOCK_UNTIL = step + max(
            1, int(STRATEGY.get("v13_gate_lock_steps", 24))
        )
    return _V13_MARKET_MODE


def _v13_senkin_action(obs, step):
    """Run the robust core plus a compatible opponent-conditioned expert."""
    copied = _copy_action(_V13_SENKIN_SCHEDULE[step])
    if _v13_market_mode(obs) != "COLLISION":
        return copied
    pipeline = _opponent_pipeline(obs)
    minimum = max(0.0, float(STRATEGY.get("v13_interference_min_exposure", 2.0)))

    def priority(pair):
        index, order = pair
        if not order or order[0] != "SELL" or len(order) < 2:
            return (1, 0.0, index)
        product = order[1]
        # Wheat is working capital for the coherent feed loop.  Its exact
        # buy/sell ordering is never changed by the market expert.
        if product == "WHEAT" or float(pipeline.get(product, 0.0)) < minimum:
            return (1, 0.0, index)
        quantity = order[2] if len(order) > 2 else 1
        return (0, -_interference_value(obs, product, quantity, pipeline), index)

    copied["market"] = [
        order for _, order in sorted(enumerate(copied["market"]), key=priority)
    ][:MAX_ORDERS]
    return copied


def _v14_market_mode(obs):
    """Select a collision expert using price-weighted public concentration.

    v13 measured concentration in physical exposure only.  v14 keeps the same
    conservative daily gate, but normalizes each visible product pipeline by
    its equilibrium price before measuring concentration.  This makes a
    premium product at the market floor weak evidence while preserving the
    signal from a genuinely valuable, concentrated pipeline.
    """
    global _V14_MARKET_MODE, _V14_MARKET_CONFIDENCE, _V14_MARKET_LOCK_UNTIL
    step = max(0, int(_get(obs, "step", 0)))
    hour = int(_get(obs, "hour", step % 24))
    if step == 0:
        _V14_MARKET_MODE = "BASE"
        _V14_MARKET_CONFIDENCE = 0.0
        _V14_MARKET_LOCK_UNTIL = -1
    if not STRATEGY.get("v14_market_adaptation", True):
        return "BASE"
    if step < _V14_MARKET_LOCK_UNTIL or (hour != 0 and step > 0):
        return _V14_MARKET_MODE

    pipeline = _opponent_pipeline(obs)
    products = ("CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
    exposures = [max(0.0, float(pipeline.get(product, 0.0))) for product in products]
    total_exposure = sum(exposures)
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    reference = {
        "CARROT": 35.0,
        "TOMATO": 60.0,
        "STRAWBERRY": 120.0,
        "MELON": 250.0,
        "EGG": 50.0,
        "MILK": 160.0,
        "WOOL": 200.0,
    }
    weighted = [
        exposure * max(1.0, float(prices.get(product, reference[product])))
        / reference[product]
        for product, exposure in zip(products, exposures)
    ]
    weighted_total = sum(weighted)
    concentration = max(weighted, default=0.0) / weighted_total if weighted_total > 0 else 0.0
    scale = max(0.1, float(STRATEGY.get("v14_gate_exposure_scale", 6.0)))
    target_concentration = max(0.1, float(STRATEGY.get("v14_gate_concentration", 0.50)))
    confidence = min(1.0, total_exposure / scale) * min(
        1.0, concentration / target_concentration
    )
    _V14_MARKET_CONFIDENCE = confidence
    threshold = max(0.0, min(1.0, float(STRATEGY.get("v14_gate_confidence", 0.70))))
    next_mode = "COLLISION" if confidence >= threshold else "BASE"
    if _V14_MARKET_MODE == "COLLISION" and confidence >= threshold * 0.5:
        next_mode = "COLLISION"
    if next_mode != _V14_MARKET_MODE:
        _V14_MARKET_MODE = next_mode
        _V14_MARKET_LOCK_UNTIL = step + max(
            1, int(STRATEGY.get("v14_gate_lock_steps", 24))
        )
    return _V14_MARKET_MODE


def _v14_senkin_action(obs, step):
    """Run the v13 route with v14's price-aware collision ordering."""
    copied = _copy_action(_V13_SENKIN_SCHEDULE[step])
    if _v14_market_mode(obs) != "COLLISION":
        return copied
    pipeline = _opponent_pipeline(obs)
    minimum = max(0.0, float(STRATEGY.get("v14_interference_min_exposure", 2.0)))

    def priority(pair):
        index, order = pair
        if not order or order[0] != "SELL" or len(order) < 2:
            return (1, 0.0, index)
        product = order[1]
        # WHEAT is the working-capital loop and remains in its validated order.
        if product == "WHEAT" or float(pipeline.get(product, 0.0)) < minimum:
            return (1, 0.0, index)
        quantity = order[2] if len(order) > 2 else 1
        return (0, -_interference_value(obs, product, quantity, pipeline), index)

    copied["market"] = [
        order for _, order in sorted(enumerate(copied["market"]), key=priority)
    ][:MAX_ORDERS]
    return copied


def _v15_market_mode(obs):
    """Use v13 exposure evidence with v14 price evidence as a consensus gate.

    The v14 leaderboard result showed that price weighting should not replace
    the physical portfolio signal.  v15 therefore activates only when both
    views agree.  The gate is never more eager than v13, while a floor-priced
    product cannot create a false collision signal by itself.
    """
    global _V15_MARKET_MODE, _V15_MARKET_CONFIDENCE, _V15_MARKET_LOCK_UNTIL
    step = max(0, int(_get(obs, "step", 0)))
    hour = int(_get(obs, "hour", step % 24))
    if step == 0:
        _V15_MARKET_MODE = "BASE"
        _V15_MARKET_CONFIDENCE = 0.0
        _V15_MARKET_LOCK_UNTIL = -1
    if not STRATEGY.get("v15_market_adaptation", True):
        return "BASE"
    if step < _V15_MARKET_LOCK_UNTIL or (hour != 0 and step > 0):
        return _V15_MARKET_MODE

    pipeline = _opponent_pipeline(obs)
    products = ("CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
    exposures = [max(0.0, float(pipeline.get(product, 0.0))) for product in products]
    total_exposure = sum(exposures)
    physical_concentration = (
        max(exposures, default=0.0) / total_exposure if total_exposure > 0 else 0.0
    )
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    reference = {
        "CARROT": 35.0,
        "TOMATO": 60.0,
        "STRAWBERRY": 120.0,
        "MELON": 250.0,
        "EGG": 50.0,
        "MILK": 160.0,
        "WOOL": 200.0,
    }
    weighted = [
        exposure * max(1.0, float(prices.get(product, reference[product])))
        / reference[product]
        for product, exposure in zip(products, exposures)
    ]
    weighted_total = sum(weighted)
    value_concentration = (
        max(weighted, default=0.0) / weighted_total if weighted_total > 0 else 0.0
    )
    scale = max(0.1, float(STRATEGY.get("v15_gate_exposure_scale", 6.0)))
    target_concentration = max(0.1, float(STRATEGY.get("v15_gate_concentration", 0.50)))
    volume_confidence = min(1.0, total_exposure / scale) * min(
        1.0, physical_concentration / target_concentration
    )
    value_confidence = min(1.0, total_exposure / scale) * min(
        1.0, value_concentration / target_concentration
    )
    # Intersection, rather than union: preserve v13's precision and use
    # v14's price view only to veto weak/floor-priced collision evidence.
    confidence = min(volume_confidence, value_confidence)
    _V15_MARKET_CONFIDENCE = confidence
    threshold = max(0.0, min(1.0, float(STRATEGY.get("v15_gate_confidence", 0.70))))
    next_mode = "COLLISION" if confidence >= threshold else "BASE"
    if _V15_MARKET_MODE == "COLLISION" and confidence >= threshold * 0.5:
        next_mode = "COLLISION"
    if next_mode != _V15_MARKET_MODE:
        _V15_MARKET_MODE = next_mode
        _V15_MARKET_LOCK_UNTIL = step + max(
            1, int(STRATEGY.get("v15_gate_lock_steps", 24))
        )
    return _V15_MARKET_MODE


def _v15_senkin_action(obs, step):
    """Run v13's route with a conservative top-five collision specialist."""
    copied = _copy_action(_V13_SENKIN_SCHEDULE[step])
    if _v15_market_mode(obs) != "COLLISION":
        return copied
    pipeline = _opponent_pipeline(obs)
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    minimum = max(0.0, float(STRATEGY.get("v15_interference_min_exposure", 2.0)))

    def priority(pair):
        index, order = pair
        if not order or order[0] != "SELL" or len(order) < 2:
            return (1, 0.0, index)
        product = order[1]
        # WHEAT remains the validated working-capital loop.  Do not spend a
        # scarce market slot on a product already pinned to the $1 floor.
        if (
            product == "WHEAT"
            or float(pipeline.get(product, 0.0)) < minimum
            or float(prices.get(product, 10 ** 9)) <= 1.0
        ):
            return (1, 0.0, index)
        quantity = order[2] if len(order) > 2 else 1
        return (0, -_interference_value(obs, product, quantity, pipeline), index)

    copied["market"] = [
        order for _, order in sorted(enumerate(copied["market"]), key=priority)
    ][:MAX_ORDERS]
    return copied


def _v16_core_schedule(obs):
    """Select one complete route by player position; never switch mid-game."""
    return (
        _V16_P0_SCHEDULE
        if int(_get(obs, "player", 0)) == 0
        else _V16_P1_SCHEDULE
    )


def _v16_market_mode(obs):
    """Choose a public-state collision lane with a slow, conservative gate."""
    global _V16_MARKET_MODE, _V16_MARKET_CONFIDENCE, _V16_MARKET_LOCK_UNTIL
    step = max(0, int(_get(obs, "step", 0)))
    hour = int(_get(obs, "hour", step % 24))
    if step == 0:
        _V16_MARKET_MODE = "BASE"
        _V16_MARKET_CONFIDENCE = 0.0
        _V16_MARKET_LOCK_UNTIL = -1
    if not STRATEGY.get("v16_market_adaptation", True):
        return "BASE"
    if step < _V16_MARKET_LOCK_UNTIL or (hour != 0 and step > 0):
        return _V16_MARKET_MODE

    pipeline = _opponent_pipeline(obs)
    products = ("CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
    exposures = [max(0.0, float(pipeline.get(product, 0.0))) for product in products]
    total_exposure = sum(exposures)
    physical_concentration = (
        max(exposures, default=0.0) / total_exposure if total_exposure > 0 else 0.0
    )
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    reference = {
        "CARROT": 35.0,
        "TOMATO": 60.0,
        "STRAWBERRY": 120.0,
        "MELON": 250.0,
        "EGG": 50.0,
        "MILK": 160.0,
        "WOOL": 200.0,
    }
    weighted = [
        exposure * max(1.0, float(prices.get(product, reference[product])))
        / reference[product]
        for product, exposure in zip(products, exposures)
    ]
    weighted_total = sum(weighted)
    value_concentration = (
        max(weighted, default=0.0) / weighted_total if weighted_total > 0 else 0.0
    )
    scale = max(0.1, float(STRATEGY.get("v16_gate_exposure_scale", 6.0)))
    target = max(0.1, float(STRATEGY.get("v16_gate_concentration", 0.50)))
    volume_confidence = min(1.0, total_exposure / scale) * min(
        1.0, physical_concentration / target
    )
    value_confidence = min(1.0, total_exposure / scale) * min(
        1.0, value_concentration / target
    )
    confidence = min(volume_confidence, value_confidence)
    top_index = max(range(len(products)), key=lambda index: weighted[index], default=0)
    top_product = products[top_index]
    top_price_ratio = max(1.0, float(prices.get(top_product, reference[top_product]))) / reference[top_product]
    if top_price_ratio < max(
        0.0, float(STRATEGY.get("v16_gate_price_floor_ratio", 0.50))
    ):
        confidence = 0.0
    _V16_MARKET_CONFIDENCE = confidence
    threshold = max(0.0, min(1.0, float(STRATEGY.get("v16_gate_confidence", 0.70))))
    if confidence >= threshold:
        margin = max(0.0, float(STRATEGY.get("v16_value_lane_margin", 0.05)))
        next_mode = (
            "VALUE" if value_confidence >= volume_confidence + margin else "VOLUME"
        )
    elif _V16_MARKET_MODE in {"VOLUME", "VALUE"} and confidence >= threshold * 0.5:
        next_mode = _V16_MARKET_MODE
    else:
        next_mode = "BASE"
    if next_mode != _V16_MARKET_MODE:
        _V16_MARKET_MODE = next_mode
        _V16_MARKET_LOCK_UNTIL = step + max(
            1, int(STRATEGY.get("v16_gate_lock_steps", 48))
        )
    return _V16_MARKET_MODE


def _v16_senkin_action(obs, step):
    """Run one coherent v16 route with a two-lane public-observation overlay."""
    copied = _copy_action(_v16_core_schedule(obs)[step])
    mode = _v16_market_mode(obs)
    if mode == "BASE":
        return copied
    pipeline = _opponent_pipeline(obs)
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    minimum = max(0.0, float(STRATEGY.get("v16_interference_min_exposure", 2.0)))

    def priority(pair):
        index, order = pair
        if not order or order[0] != "SELL" or len(order) < 2:
            return (1, 0.0, index)
        product = order[1]
        # WHEAT is working capital; its exact position is never reordered.
        if (
            product == "WHEAT"
            or float(pipeline.get(product, 0.0)) < minimum
            or float(prices.get(product, 10 ** 9)) <= 1.0
        ):
            return (1, 0.0, index)
        quantity = order[2] if len(order) > 2 else 1
        exposure = max(0.0, float(pipeline.get(product, 0.0)))
        score = (
            _interference_value(obs, product, quantity, pipeline)
            if mode == "VALUE"
            else exposure * min(float(quantity or 1), 10.0)
        )
        return (0, -score, index)

    copied["market"] = [
        order for _, order in sorted(enumerate(copied["market"]), key=priority)
    ][:MAX_ORDERS]
    return copied


def _v17_number(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _v17_clip(value, low=-20.0, high=20.0):
    return min(high, max(low, _v17_number(value)))


def _v17_ready_amount(tile):
    for key in ("yield_units", "ready_yield", "yield", "amount", "quantity"):
        if key in tile:
            return max(0.0, _v17_number(tile.get(key)))
    return 1.0 if tile.get("ready") is True or tile.get("is_ready") is True else 0.0


def _v17_public_farm_stats(farm):
    """Mirror the offline public-only feature extractor without private data."""
    products = tuple(_V17_MARKET_MODEL["products"])
    supply = {product: 0.0 for product in products}
    ready = {product: 0.0 for product in products}
    crop_to_product = {
        product: product for product in ("CARROT", "TOMATO", "STRAWBERRY", "MELON")
    }
    animal_to_product = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
    for row in (_get(farm, "tiles", []) or []):
        row = row if isinstance(row, list) else [row]
        for tile in row:
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            animal = tile.get("animal")
            product = crop_to_product.get(str(crop).upper()) if crop is not None else None
            if product is None and animal is not None:
                product = animal_to_product.get(str(animal).upper())
            if product is None:
                product = animal_to_product.get(str(tile.get("kind", "")).upper())
            if product is None:
                continue
            amount = _v17_ready_amount(tile)
            ready[product] += amount
            supply[product] += 1.0 + amount
    return {"supply": supply, "ready": ready}


def _v17_candidate_features(obs, product, planned_quantity):
    """Return the frozen model's 53 public candidate features in schema order."""
    products = tuple(_V17_MARKET_MODEL["products"])
    if product not in products:
        return None
    farms = _get(obs, "farms", []) or []
    player = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
    own_farm = farms[player] if player < len(farms) else {}
    opponent_farm = farms[1 - player] if len(farms) > 1 else {}
    own = _v17_public_farm_stats(own_farm)
    opponent = _v17_public_farm_stats(opponent_farm)
    step = max(0.0, _v17_number(_get(obs, "step", 0)))
    day = max(0.0, _v17_number(_get(obs, "day", math.floor(step / 24.0))))
    hour = _v17_number(_get(obs, "hour", step % 24.0)) % 24.0
    market = _get(obs, "market", {}) or {}
    prices_raw = _get(market, "prices", None)
    if not isinstance(prices_raw, dict):
        prices_raw = _get(market, "current_prices", {}) or {}
    prices = {
        str(key).upper(): max(0.0, _v17_number(value))
        for key, value in prices_raw.items()
    }
    price_values = [max(1.0, prices.get(candidate, 1.0)) for candidate in products]
    mean_price = sum(price_values) / len(price_values)
    max_price = max(price_values)
    total_opponent_supply = sum(opponent["supply"].values())
    total_opponent_value = sum(
        opponent["supply"].get(candidate, 0.0)
        * max(1.0, prices.get(candidate, 1.0))
        for candidate in products
    )
    index = products.index(product)
    onehot = [0.0] * len(products)
    onehot[index] = 1.0
    day_fraction = min(1.0, day / 30.0)
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)
    values = list(onehot)
    values.extend(value * day_fraction for value in onehot)
    values.extend(value * float(player) for value in onehot)
    values.extend(value * hour_sin for value in onehot)
    values.extend(value * hour_cos for value in onehot)
    price = max(1.0, prices.get(product, 1.0))
    own_supply = own["supply"].get(product, 0.0)
    opponent_supply = opponent["supply"].get(product, 0.0)
    opponent_value = opponent_supply * price
    own_value = own_supply * price
    own_total_value = sum(
        own["supply"].get(candidate, 0.0) * max(1.0, prices.get(candidate, 1.0))
        for candidate in products
    )
    price_rank = 1.0 + sum(other < price for other in price_values)
    quantity = max(0.0, _v17_number(planned_quantity))
    values.extend((
        math.log1p(quantity),
        min(1.0, quantity / 20.0),
        math.log1p(price),
        _v17_clip(math.log(price / max(1.0, mean_price))),
        _v17_clip(math.log(price / max(1.0, max_price))),
        price_rank / len(products),
        math.log1p(max(0.0, own_supply)),
        math.log1p(max(0.0, opponent_supply)),
        math.log1p(max(0.0, own["ready"].get(product, 0.0))),
        math.log1p(max(0.0, opponent["ready"].get(product, 0.0))),
        opponent_supply / max(1.0, total_opponent_supply),
        opponent_value / max(1.0, total_opponent_value),
        own_value / max(1.0, own_total_value),
    ))
    return [_v17_clip(value) for value in values]


def _v17_pair_probability(obs, left_order, right_order):
    """Predict which of two distinct product orders should execute first."""
    products = tuple(_V17_MARKET_MODEL["products"])
    left_product, right_product = left_order[1], right_order[1]
    if left_product == right_product:
        return 0.5
    left_quantity = left_order[2] if len(left_order) > 2 else 1
    right_quantity = right_order[2] if len(right_order) > 2 else 1
    # Training labels use the canonical product order.  Orient the runtime pair
    # the same way, then flip the probability back to the caller's order.
    caller_is_canonical = products.index(left_product) < products.index(right_product)
    canonical_left = left_order if caller_is_canonical else right_order
    canonical_right = right_order if caller_is_canonical else left_order
    canonical_left_quantity = left_quantity if caller_is_canonical else right_quantity
    canonical_right_quantity = right_quantity if caller_is_canonical else left_quantity
    left_features = _v17_candidate_features(
        obs, canonical_left[1], canonical_left_quantity
    )
    right_features = _v17_candidate_features(
        obs, canonical_right[1], canonical_right_quantity
    )
    if left_features is None or right_features is None:
        return 0.5
    standardization = _V17_MARKET_MODEL["standardization"]
    mean = standardization["mean"]
    scale = standardization["scale"]
    difference = [left - right for left, right in zip(left_features, right_features)]
    standardized = [
        (value - center) / (spread if abs(spread) > 1e-12 else 1.0)
        for value, center, spread in zip(difference, mean, scale)
    ]
    layers = _V17_MARKET_MODEL["layers"]
    hidden = []
    for hidden_index, bias in enumerate(layers["hidden_bias"]):
        total = bias + sum(
            value * layers["input_to_hidden"][feature_index][hidden_index]
            for feature_index, value in enumerate(standardized)
        )
        hidden.append(math.tanh(total))
    logit = layers["output_bias"] + sum(
        value * weight for value, weight in zip(hidden, layers["hidden_to_output"])
    )
    logit /= max(1e-6, float(_V17_MARKET_MODEL["calibration_temperature"]))
    if logit >= 0.0:
        probability = 1.0 / (1.0 + math.exp(-min(60.0, logit)))
    else:
        exponential = math.exp(max(-60.0, logit))
        probability = exponential / (1.0 + exponential)
    return probability if caller_is_canonical else 1.0 - probability


def _v17_learned_action(obs, step):
    """Re-rank only existing non-WHEAT SELLs; keep every protected slot fixed."""
    copied = _copy_action(_V17_SCHEDULE[step])
    if not STRATEGY.get("v17_market_ranker", True):
        return copied
    products = set(_V17_MARKET_MODEL["products"])
    free_indices = [
        index
        for index, order in enumerate(copied["market"])
        if order and len(order) >= 2 and order[0] == "SELL" and order[1] in products
    ]
    if len(free_indices) < 2:
        return copied
    scores = {index: 0.0 for index in free_indices}
    minimum_confidence = max(
        0.0, min(1.0, float(STRATEGY.get("v17_rank_min_confidence", 0.0)))
    )
    for offset, left_index in enumerate(free_indices):
        for right_index in free_indices[offset + 1:]:
            probability = _v17_pair_probability(
                obs, copied["market"][left_index], copied["market"][right_index]
            )
            if 2.0 * abs(probability - 0.5) < minimum_confidence:
                probability = 0.5
            scores[left_index] += probability
            scores[right_index] += 1.0 - probability
    ranked_orders = [
        copied["market"][index]
        for index in sorted(free_indices, key=lambda index: (-scores[index], index))
    ]
    for index, order in zip(free_indices, ranked_orders):
        copied["market"][index] = order
    return copied


_V18_PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)


def _v18_state_features(obs):
    """Public own-state vector used by the offline and submission gates."""
    player = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
    farms = _get(obs, "farms", []) or []
    farm = farms[player] if player < len(farms) and isinstance(farms[player], dict) else {}
    private = _get(obs, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    market = _get(obs, "market", {}) or {}
    prices = _get(market, "prices", _get(market, "current_prices", {})) or {}
    counts = {
        name: 0.0
        for name in (
            "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "COW", "SHEEP", "GOOSE",
        )
    }
    for row in farm.get("tiles", []) or []:
        for tile in row if isinstance(row, list) else [row]:
            if not isinstance(tile, dict):
                continue
            crop = str(tile.get("crop", "")).upper()
            animal = str(tile.get("animal", tile.get("kind", ""))).upper()
            if crop in counts:
                counts[crop] += 1.0
            if animal in counts:
                counts[animal] += 1.0
    values = [
        math.log1p(max(0.0, _v17_number(farm.get("money", 0)))),
        len(farm.get("hands", []) or []) / 16.0,
        len(farm.get("unlocked_quadrants", []) or []) / 4.0,
    ]
    values.extend(
        counts[name] / 50.0
        for name in (
            "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "COW", "SHEEP", "GOOSE",
        )
    )
    values.extend(
        math.log1p(max(0.0, _v17_number(shed.get(name, 0))))
        for name in _V18_PRODUCTS
    )
    price_values = [
        max(1.0, _v17_number(prices.get(name, 1), 1.0))
        for name in _V18_PRODUCTS
    ]
    mean_price = sum(price_values) / len(price_values)
    values.extend(math.log(value / mean_price) for value in price_values)
    return values


def _v18_closed_loop_action(obs, step):
    """Lock a seat route and choose a complete market expert once per day.

    The learned choice is outcome-based: complete-game wins set the seat
    priors, and public state similarity supplies the closed-loop correction.
    No SELL-order imitation label is used at training or runtime.
    """
    global _V18_SELECTED_MARKET, _V18_SELECTED_DAY, _V18_SELECTED_BOARD
    seat = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
    experts = _V18_RUNTIME["experts"]
    base_board_name = _V18_RUNTIME["board_by_seat"][str(seat)]
    base_board_actions = experts[base_board_name]["actions"]
    bounded_step = min(max(0, int(step)), len(base_board_actions) - 1)
    if bounded_step == 0:
        _V18_SELECTED_MARKET[seat] = None
        _V18_SELECTED_DAY[seat] = None
        _V18_SELECTED_BOARD[seat] = None

    board_strength = float(_V18_RUNTIME.get("board_distance_strength", 0.0))
    board_fork_step = int(_V18_RUNTIME.get("board_fork_step", len(base_board_actions)))
    if (
        STRATEGY.get("v18_closed_loop_board", True)
        and board_strength > 0.0
        and bounded_step >= board_fork_step
        and _V18_SELECTED_BOARD[seat] is None
    ):
        current = _v18_state_features(obs)
        scales = _V18_RUNTIME["feature_standardization"]["scale"]
        bias = _V18_RUNTIME["board_bias_by_seat"][str(seat)]
        best_board = None
        for name, expert in experts.items():
            prototype = expert["board_prototype_at_fork"]
            distance = sum(
                ((value - center) / max(1e-12, float(scale))) ** 2
                for value, center, scale in zip(current, prototype, scales)
            ) / len(current)
            candidate = (float(bias.get(name, 0.0)) - board_strength * distance, name)
            if best_board is None or candidate > best_board:
                best_board = candidate
        _V18_SELECTED_BOARD[seat] = best_board[1]

    board_name = _V18_SELECTED_BOARD[seat] or base_board_name
    board_actions = experts[board_name]["actions"]
    board_action = board_actions[bounded_step] or {
        "farmer": ["PASS"], "hands": [], "market": [],
    }
    if not STRATEGY.get("v18_closed_loop_market", True):
        return _copy_action(board_action)

    day = max(0, int(_get(obs, "day", bounded_step // 24) or 0))
    if _V18_SELECTED_DAY[seat] != day or _V18_SELECTED_MARKET[seat] is None:
        current = _v18_state_features(obs)
        scales = _V18_RUNTIME["feature_standardization"]["scale"]
        bias = _V18_RUNTIME["market_bias_by_seat"][str(seat)]
        distance_strength = float(_V18_RUNTIME["distance_strength"])
        stay_bonus = float(_V18_RUNTIME["stay_bonus"])
        selected = _V18_SELECTED_MARKET[seat]
        best = None
        for name, expert in experts.items():
            prototypes = expert["prototypes_by_day"]
            prototype = prototypes[min(day, len(prototypes) - 1)]
            distance = sum(
                ((value - center) / max(1e-12, float(scale))) ** 2
                for value, center, scale in zip(current, prototype, scales)
            ) / len(current)
            score = float(bias.get(name, 0.0)) - distance_strength * distance
            if name == selected:
                score += stay_bonus
            candidate = (score, name)
            if best is None or candidate > best:
                best = candidate
        _V18_SELECTED_MARKET[seat] = best[1]
        _V18_SELECTED_DAY[seat] = day

    market_name = _V18_SELECTED_MARKET[seat]
    market_actions = experts[market_name]["actions"]
    market_action = market_actions[min(bounded_step, len(market_actions) - 1)] or {}
    return {
        "farmer": list(board_action.get("farmer") or ["PASS"]),
        "hands": [list(order) for order in (board_action.get("hands") or [])],
        "market": [list(order) for order in (market_action.get("market") or [])],
    }


def _public_farm_counts(farm):
    """Return stable public portfolio features for a farm.

    Fixed replay executors cannot safely swap movement trajectories halfway
    through a game.  Animal species and within-turn purchase priority are
    different: both share the same pasture sites and unit actions, so they can
    react to public state without invalidating the remaining executor.
    """
    counts = {
        "COW": 0,
        "SHEEP": 0,
        "GOOSE": 0,
        "PLANTS": 0,
        "STRAWBERRY": 0,
        "LAND": len(_get(farm, "unlocked_quadrants", []) or []),
        "MONEY": float(_get(farm, "money", 0) or 0),
    }
    for row in (_get(farm, "tiles", []) or []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            animal = tile.get("animal")
            if animal in ("COW", "SHEEP", "GOOSE"):
                counts[animal] += 1
            if tile.get("kind") == "PLANT":
                counts["PLANTS"] += 1
                if tile.get("crop") == "STRAWBERRY":
                    counts["STRAWBERRY"] += 1
    counts["ANIMALS"] = counts["COW"] + counts["SHEEP"] + counts["GOOSE"]
    return counts


def _adaptive_animal_focus(obs, own, opponent):
    """Choose a livestock market to contest from observable exposure only."""
    day = int(_get(obs, "day", 0))
    if not (
        int(STRATEGY.get("adaptive_animal_min_day", 2))
        <= day
        <= int(STRATEGY.get("adaptive_animal_max_day", 14))
    ):
        return None
    opponent_herd = opponent["COW"] + opponent["SHEEP"]
    if opponent_herd < int(STRATEGY.get("adaptive_animal_min_herd", 4)):
        return None
    lead = int(STRATEGY.get("adaptive_animal_lead", 2))
    if opponent["COW"] >= opponent["SHEEP"] + lead:
        focus = "COW"
    elif opponent["SHEEP"] >= opponent["COW"] + lead:
        focus = "SHEEP"
    elif STRATEGY.get("adaptive_tempo_cow") and (
        opponent["ANIMALS"] >= own["ANIMALS"] + int(
            STRATEGY.get("adaptive_tempo_animal_lead", 1)
        )
        or opponent["LAND"] >= own["LAND"] + int(
            STRATEGY.get("adaptive_tempo_land_lead", 1)
        )
    ):
        # Cows are the cheaper livestock asset and start the milk cycle sooner.
        # When a balanced rival is already ahead on capital, this is a recovery
        # branch rather than an attempt to infer a nonexistent specialization.
        focus = "COW"
    else:
        return None
    if STRATEGY.get("adaptive_animal_mode") == "diversify":
        focus = "SHEEP" if focus == "COW" else "COW"
    share = max(0.5, min(1.0, float(STRATEGY.get("adaptive_animal_target_share", 0.72))))
    own_herd = own["COW"] + own["SHEEP"]
    # Do not blindly convert every future purchase.  Stop contesting once the
    # requested share is already represented in our live herd.
    target = int(round((own_herd + 1) * share))
    return focus if own[focus] < target else None


def _prioritize_capital_orders(obs, orders, own, opponent):
    """Spend existing early orders sooner when the rival is accelerating.

    WHEAT orders keep their exact slots and relative order because the fixed
    executor uses them as a cash cycle.  Only non-WHEAT slots are permuted.
    """
    if not STRATEGY.get("adaptive_capital_priority"):
        return orders
    if int(_get(obs, "day", 0)) > int(STRATEGY.get("adaptive_capital_max_day", 12)):
        return orders
    animal_pressure = opponent["ANIMALS"] >= own["ANIMALS"] + int(
        STRATEGY.get("adaptive_capital_animal_lead", 2)
    )
    land_pressure = opponent["LAND"] >= own["LAND"] + int(
        STRATEGY.get("adaptive_capital_land_lead", 1)
    )
    if not (animal_pressure or land_pressure):
        return orders

    movable = []
    positions = []
    for index, order in enumerate(orders):
        if len(order) > 1 and order[1] == "WHEAT" and order[0] in {"BUY_PRODUCT", "SELL"}:
            continue
        positions.append(index)
        movable.append(order)

    def priority(pair):
        index, order = pair
        command = order[0] if order else ""
        if command == "SELL":
            return (0, index)
        if command in {"BUY_LAND", "BUY_ANIMAL"}:
            return (1, index)
        if command == "HIRE":
            return (2, index)
        return (3, index)

    reordered = list(orders)
    sorted_orders = [order for _, order in sorted(enumerate(movable), key=priority)]
    for index, order in zip(positions, sorted_orders):
        reordered[index] = order
    return reordered


def _apply_fixed_board_adaptation(obs, action):
    """Observation-only adaptation layered on a validated fixed executor."""
    copied = _copy_action(action)
    if not STRATEGY.get("fixed_board_adaptation"):
        return copied
    farms = _get(obs, "farms", []) or []
    player = int(_get(obs, "player", 0))
    if len(farms) != 2 or player not in (0, 1):
        return copied
    own = _public_farm_counts(farms[player])
    opponent = _public_farm_counts(farms[1 - player])
    focus = _adaptive_animal_focus(obs, own, opponent)
    if focus:
        for order in copied["market"]:
            if order and order[0] == "BUY_ANIMAL" and len(order) >= 2:
                order[1] = focus
    copied["market"] = _prioritize_capital_orders(obs, copied["market"], own, opponent)[:MAX_ORDERS]
    return copied


def _v11_radiant_schedule(obs, step):
    """Lock one coherent radiant trajectory from a public day-four price.

    The robust and alpha trajectories share actions 0..108 exactly.  Routing
    at step 109 therefore changes no prior farm state, and locking the choice
    prevents the invalid cross-trajectory oscillation seen in k-NN ablations.
    """
    global _V11_SELECTED_RADIANT_VARIANT
    variant = STRATEGY.get("v11_radiant_variant", "robust")
    if step == 0:
        _V11_SELECTED_RADIANT_VARIANT = None
    if variant in {"robust", "alpha"}:
        selected = variant
    else:
        route_step = int(STRATEGY.get("v11_route_step", 109))
        if step >= route_step and _V11_SELECTED_RADIANT_VARIANT is None:
            prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
            milk_price = float(prices.get("MILK", 0) or 0)
            _V11_SELECTED_RADIANT_VARIANT = (
                "alpha"
                if milk_price >= float(STRATEGY.get("v11_alpha_milk_price", 193))
                else "robust"
            )
        selected = _V11_SELECTED_RADIANT_VARIANT or "robust"
    return _V11_RADIANT_ALPHA_SCHEDULE if selected == "alpha" else _V11_RADIANT_SCHEDULE


def _v12_syouya_action(obs, step):
    """Apply only the two validated late gates to the coherent syouya route.

    Episodes 89511601 and 89512693 have identical movement and capital plans.
    Their first difference is the order of the terminal MILK/STRAWBERRY sales;
    public production capacity times the current price selects that order.
    """
    copied = _copy_action(_V12_SYOUYA_SCHEDULE[step])
    mode = STRATEGY.get("v12_late_market_mode", "price")
    if step == 624 and mode in {"asset", "price"}:
        farms = _get(obs, "farms", []) or []
        player = int(_get(obs, "player", 0))
        counts = (
            _public_farm_counts(farms[1 - player])
            if len(farms) == 2 and player in (0, 1)
            else {"COW": 0, "STRAWBERRY": 0}
        )
        if mode == "asset":
            milk_first = counts["COW"] >= counts["STRAWBERRY"]
        else:
            prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
            milk_value = counts["COW"] * float(prices.get("MILK", 0) or 0)
            strawberry_value = counts["STRAWBERRY"] * float(
                prices.get("STRAWBERRY", 0) or 0
            )
            milk_first = milk_value >= strawberry_value
        if not milk_first:
            priority = {
                "STRAWBERRY": 0,
                "MELON": 1,
                "FERTILIZER": 2,
                "MILK": 3,
                "WHEAT": 4,
            }
            sales = [order for order in copied["market"] if order and order[0] == "SELL"]
            other = [order for order in copied["market"] if not order or order[0] != "SELL"]
            copied["market"] = sorted(
                sales, key=lambda order: priority.get(order[1], len(priority))
            ) + other
    elif step == 714:
        shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
        if float(_get(shed, "FERTILIZER", 0) or 0) >= 2:
            copied["market"] = (copied["market"] + [["SELL", "FERTILIZER", 2]])[:MAX_ORDERS]
    return copied


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _shed_access(board_size):
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _available_access(tiles):
    """Shed corners that belong to currently unlocked quadrants."""
    access = _shed_access(len(tiles) or 10)
    available = tuple(p for p in access if tiles[p[1]][p[0]] != "LOCKED") if tiles else ()
    return available or (access[0],)


def _move_toward(pos, target, tiles=None):
    """Return a shortest move while avoiding still-locked quadrants."""
    x, y = pos
    tx, ty = target
    if tiles:
        moves = (("NORTH", 0, -1), ("WEST", -1, 0), ("EAST", 1, 0), ("SOUTH", 0, 1))
        queue = [(x, y, None)]
        seen = {(x, y)}
        for cx, cy, first in queue:
            if (cx, cy) == (tx, ty):
                return [first] if first else ["PASS"]
            for name, dx, dy in moves:
                nx, ny = cx + dx, cy + dy
                if not (0 <= ny < len(tiles) and 0 <= nx < len(tiles[ny])) or (nx, ny) in seen:
                    continue
                if tiles[ny][nx] == "LOCKED":
                    continue
                seen.add((nx, ny))
                queue.append((nx, ny, first or name))
        return ["PASS"]
    if abs(tx - x) >= abs(ty - y) and x != tx:
        return ["EAST" if x < tx else "WEST"]
    if y != ty:
        return ["SOUTH" if y < ty else "NORTH"]
    if x != tx:
        return ["EAST" if x < tx else "WEST"]
    return ["PASS"]


def _count_inventory(inv):
    if not isinstance(inv, dict):
        return 0
    return sum(max(0, int(v)) for v in inv.values())


def _asset_counts(obs):
    player = int(_get(obs, "player", 0))
    farm = _get(obs, "farms", [])[player]
    private = _get(obs, "private", {}) or {}
    counts = {name: 0 for name in ANIMALS}
    for row in _get(farm, "tiles", []):
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in counts:
                counts[tile["animal"]] += 1
    shed = _get(private, "shed", {}) or {}
    inventories = _get(private, "inventories", []) or []
    for animal in counts:
        counts[animal] += int(shed.get(animal, 0))
        counts[animal] += sum(int(inv.get(animal, 0)) for inv in inventories if isinstance(inv, dict))
    return counts


def _active_target(pos, day, unlocked):
    x, y = pos
    if x < 5 and y < 5:
        return True
    if x >= 5 and y < 5:
        return "NE" in unlocked and day >= 7
    if x < 5 and y >= 5:
        return "SW" in unlocked and day >= 9
    return False


def _animal_site_active(pos, day, unlocked):
    """Stage livestock growth so labour and feed can grow before the herd."""
    try:
        index = ANIMAL_SITES.index(pos)
    except ValueError:
        return False
    opening_animals = min(4, max(0, int(STRATEGY.get("opening_animals", 2))))
    if index < opening_animals:
        return True
    if index < 4:
        return day >= int(STRATEGY.get("animal_nw_day", 4))
    if index < 9:
        return day >= int(STRATEGY.get("animal_ne_day", 8)) and "NE" in unlocked
    return day >= int(STRATEGY.get("animal_sw_day", 12)) and "SW" in unlocked


def _crop_is_ripe(tile, day, hour):
    crop = tile.get("crop")
    spec = CROPS.get(crop)
    if not spec or int(tile.get("yield_units", 0)) <= 0:
        return False
    age = day - int(tile.get("planted_day", day))
    if age < spec["first"]:
        return False
    if not spec["ongoing"]:
        return int(tile.get("yield_units", 0)) >= spec["max_yield"] or age >= spec["max_day"]
    # Avoid hitting the held-yield cap, and cash out anything available near
    # the end of the season.
    threshold = max(1, int(STRATEGY.get("ongoing_harvest_threshold", 3)))
    return (
        int(tile.get("yield_units", 0)) >= threshold
        or day >= 28
        or (hour >= 18 and int(tile.get("yield_units", 0)) >= min(2, threshold))
    )


def _last_plant(crop):
    if crop == "STRAWBERRY":
        return int(_style_setting("strawberry_last_plant"))
    return int(CROPS[crop]["last_plant"])


def _fertilizer_value(tile, day, prices):
    """Expected value of one 3-day fertilizer application on a strawberry."""
    roi = STRATEGY.get("fertilizer_roi")
    if roi is None or tile.get("crop") != "STRAWBERRY":
        return 0
    if int(tile.get("fertilized_until_day", -1)) >= day + 1:
        return 0
    planted = int(tile.get("planted_day", day))
    bonus_ticks = 0
    for current_day in range(day, day + 3):
        since_first = current_day + 1 - planted - CROPS["STRAWBERRY"]["first"]
        if since_first >= 0 and since_first % 2 == 0 and since_first // 2 < CROPS["STRAWBERRY"]["max_yield"]:
            bonus_ticks += 1
    value = bonus_ticks * float(prices.get("STRAWBERRY", 120))
    cost = float(prices.get("FERTILIZER", 100)) * float(roi)
    return bonus_ticks if bonus_ticks and value >= cost else 0


def _fertilizer_positions(obs):
    player = int(_get(obs, "player", 0))
    farm = _get(obs, "farms", [])[player]
    day = int(_get(obs, "day", 0))
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    positions = []
    for y, row in enumerate(_get(farm, "tiles", [])):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PLANT" and _fertilizer_value(tile, day, prices):
                positions.append((x, y))
    return positions


def _task(priority, pos, action, requirement=None, tag=""):
    return (priority, pos, action, requirement, tag)


def _build_tasks(obs, positions, inventories):
    player = int(_get(obs, "player", 0))
    farm = _get(obs, "farms", [])[player]
    tiles = _get(farm, "tiles", [])
    private = _get(obs, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    seeds = _get(private, "seeds", {}) or {}
    day = int(_get(obs, "day", 0))
    hour = int(_get(obs, "hour", 0))
    unlocked = set(_get(farm, "unlocked_quadrants", ["NW"]) or ["NW"])
    access = _available_access(tiles)
    tasks = []
    crop_plan = _crop_plan(day)
    animal_plan = _animal_plan()
    fertilizer_positions = set(_fertilizer_positions(obs))

    animals = []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            if tile.get("animal") in ANIMALS:
                animals.append(((x, y), tile))
                if day < 29 and not tile.get("fed_today", False):
                    urgent = 0 if int(tile.get("consecutive_unfed", 0)) >= 1 or hour >= 15 else 2
                    tasks.append(_task(urgent, (x, y), ["FEED"], "WHEAT", "feed"))
                if int(tile.get("yield_units", 0)) > 0:
                    tasks.append(_task(1, (x, y), ["HARVEST"], None, "harvest"))
                if not tile.get("cared_today", False) and day < 29:
                    tasks.append(_task(3, (x, y), ["CARE"], None, "care"))
                if tile.get("fertilizer_available", False):
                    tasks.append(_task(4, (x, y), ["COLLECT_FERTILIZER"], None, "fertilizer"))

    # Crop preservation and harvest precede new construction.
    for (x, y), desired in crop_plan.items():
        if y >= len(tiles) or x >= len(tiles[y]) or not _active_target((x, y), day, unlocked):
            continue
        tile = tiles[y][x]
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if _crop_is_ripe(tile, day, hour):
                tasks.append(_task(1, (x, y), ["HARVEST"], None, "harvest"))
            elif not tile.get("watered_today", False):
                crop = tile.get("crop")
                age = day - int(tile.get("planted_day", day))
                spec = CROPS.get(crop, CROPS["WHEAT"])
                in_bonus = not spec["ongoing"] and (spec["max_day"] + 1) // 2 <= age <= spec["max_day"]
                urgent = int(tile.get("consecutive_unwatered", 0)) >= 1 or hour >= 16
                # Ongoing crops need only alternate-day watering; one-time crop
                # bonus windows are watered every day.
                needs_fertilizer_water = (x, y) in fertilizer_positions or int(tile.get("fertilized_until_day", -1)) >= day
                if urgent or in_bonus or needs_fertilizer_water:
                    tasks.append(_task(0 if urgent else 2 if needs_fertilizer_water else 3, (x, y), ["WATER"], None, "water"))
            if (x, y) in fertilizer_positions:
                tasks.append(_task(2, (x, y), ["FERTILIZE"], "FERTILIZER", "fertilize"))

    # Build and populate livestock sites before filling expansion crops.
    for pos, animal in animal_plan.items():
        x, y = pos
        if y >= len(tiles) or x >= len(tiles[y]) or not _animal_site_active(pos, day, unlocked):
            continue
        tile = tiles[y][x]
        if tile is None:
            tasks.append(_task(5, pos, ["BUILD_PASTURE"], None, "build"))
        elif isinstance(tile, dict) and tile.get("kind") == "PASTURE" and "animal" not in tile:
            tasks.append(_task(1, pos, ["PLACE", animal], animal, "place"))
        elif isinstance(tile, dict) and tile.get("kind") in ("WEED", "PLANT"):
            tasks.append(_task(5, pos, ["DIG"], None, "dig"))

    # Land preparation and planting.  Seed demand is capped here so atomic
    # validation cannot cancel several simultaneous PLANT actions.
    remaining = {crop: int(seeds.get(crop, 0)) for crop in CROPS}
    for pos, crop in crop_plan.items():
        x, y = pos
        if y >= len(tiles) or x >= len(tiles[y]) or not _active_target(pos, day, unlocked):
            continue
        tile = tiles[y][x]
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            if day <= _last_plant(crop):
                tasks.append(_task(6, pos, ["DIG"], None, "dig"))
        elif tile is None and day <= _last_plant(crop) and remaining[crop] > 0:
            tasks.append(_task(7, pos, ["PLANT", crop], None, "plant"))
            remaining[crop] -= 1

    # Operational inventory pickups.  A few loaded units can visit several
    # animals, which is much cheaper than sending every hand back to the shed.
    unfed = sum(not tile.get("fed_today", False) for _, tile in animals)
    carried_wheat = sum(int(inv.get("WHEAT", 0)) for inv in inventories if isinstance(inv, dict))
    if unfed > carried_wheat and int(shed.get("WHEAT", 0)) > 0:
        carriers = min(3, max(1, (unfed - carried_wheat + 3) // 4))
        quantity = min(int(shed.get("WHEAT", 0)), max(1, (unfed - carried_wheat + carriers - 1) // carriers))
        for i in range(carriers):
            tasks.append(_task(1, access[i % len(access)], ["PICKUP", "WHEAT", quantity], None, "pickup_wheat"))

    carried_fertilizer = sum(int(inv.get("FERTILIZER", 0)) for inv in inventories if isinstance(inv, dict))
    fertilizer_deficit = len(fertilizer_positions) - carried_fertilizer
    if fertilizer_deficit > 0 and int(shed.get("FERTILIZER", 0)) > 0:
        carriers = min(3, max(1, (fertilizer_deficit + 3) // 4))
        quantity = min(int(shed.get("FERTILIZER", 0)), max(1, (fertilizer_deficit + carriers - 1) // carriers))
        for i in range(carriers):
            tasks.append(_task(1, access[-(i % len(access)) - 1], ["PICKUP", "FERTILIZER", quantity], None, "pickup_fertilizer"))

    for animal in ANIMALS:
        empty_positions = [
            pos for pos, target in animal_plan.items()
            if target == animal and _active_target(pos, day, unlocked)
            and isinstance(tiles[pos[1]][pos[0]], dict)
            and tiles[pos[1]][pos[0]].get("kind") == "PASTURE"
            and "animal" not in tiles[pos[1]][pos[0]]
        ]
        empty = len(empty_positions)
        carried = sum(int(inv.get(animal, 0)) for inv in inventories if isinstance(inv, dict))
        if empty > carried and int(shed.get(animal, 0)) > 0:
            pickup = min(access, key=lambda a: (min(_distance(a, target) for target in empty_positions), a[1], a[0]))
            tasks.append(_task(1, pickup, ["PICKUP", animal, min(empty - carried, int(shed.get(animal, 0)))], None, "pickup_animal"))

    return tasks


def _eligible(task, inv):
    requirement = task[3]
    return requirement is None or (isinstance(inv, dict) and int(inv.get(requirement, 0)) > 0)


def _quadrant(pos):
    x, y = pos
    return "NW" if x < 5 and y < 5 else "NE" if y < 5 else "SW" if x < 5 else "SE"


def _worker_zone(index, unlocked):
    if "SW" in unlocked:
        return "NW" if index < 4 else "NE" if index < 9 else "SW"
    if "NE" in unlocked:
        return "NW" if index < 4 else "NE"
    return "NW"


def _assign_actions(obs):
    player = int(_get(obs, "player", 0))
    farm = _get(obs, "farms", [])[player]
    private = _get(obs, "private", {}) or {}
    positions = [tuple(_get(farm, "farmer", (4, 4)))] + [tuple(p) for p in (_get(farm, "hands", []) or [])]
    inventories = list(_get(private, "inventories", []) or [])
    while len(inventories) < len(positions):
        inventories.append({})
    day = int(_get(obs, "day", 0))
    hour = int(_get(obs, "hour", 0))
    access = _available_access(_get(farm, "tiles", []))
    tasks = _build_tasks(obs, positions, inventories)
    actions = [["PASS"] for _ in positions]
    free = set(range(len(positions)))

    # Purchased livestock is high-value, per-unit inventory.  Route carriers
    # directly instead of letting generic nearby tasks repeatedly steal them.
    tiles = _get(farm, "tiles", [])
    unlocked = set(_get(farm, "unlocked_quadrants", ["NW"]) or ["NW"])
    animal_plan = _animal_plan()
    reserved_targets = set()

    # Feeding is an existential task: two unfed days delete the animal.  Keep
    # designated carriers moving to the shed instead of allowing generic
    # nearest-task matching to redirect them to watering on every turn.
    if day < 29:
        unfed = [
            (x, y)
            for y, row in enumerate(tiles)
            for x, tile in enumerate(row)
            if isinstance(tile, dict)
            and tile.get("animal") in ANIMALS
            and not tile.get("fed_today", False)
        ]
        carried_wheat = sum(int(inv.get("WHEAT", 0)) for inv in inventories if isinstance(inv, dict))
        shed = _get(private, "shed", {}) or {}
        deficit = max(0, len(unfed) - carried_wheat)
        if deficit and int(shed.get("WHEAT", 0)) > 0:
            carriers = min(3, max(1, (deficit + 3) // 4), len(free))
            candidates = [
                idx for idx in free
                if isinstance(inventories[idx], dict)
                and not any(int(inventories[idx].get(a, 0)) > 0 for a in ANIMALS)
                and int(inventories[idx].get("WHEAT", 0)) == 0
            ]
            candidates.sort(key=lambda idx: min(_distance(positions[idx], p) for p in access))
            remaining_wheat = min(deficit, int(shed.get("WHEAT", 0)))
            for number, idx in enumerate(candidates[:carriers]):
                remaining_carriers = carriers - number
                quantity = max(1, (remaining_wheat + remaining_carriers - 1) // remaining_carriers)
                target = min(access, key=lambda p: (_distance(positions[idx], p), p[1], p[0]))
                actions[idx] = ["PICKUP", "WHEAT", quantity] if positions[idx] in access else _move_toward(positions[idx], target, tiles)
                remaining_wheat = max(0, remaining_wheat - quantity)
                free.discard(idx)

    for idx, (pos, inv) in enumerate(zip(positions, inventories)):
        if idx not in free:
            continue
        if not isinstance(inv, dict):
            continue
        animal = next((name for name in ANIMALS if int(inv.get(name, 0)) > 0), None)
        if animal is None:
            continue
        targets = [
            target
            for target, desired in animal_plan.items()
            if desired == animal
            and target not in reserved_targets
            and _active_target(target, day, unlocked)
            and isinstance(tiles[target[1]][target[0]], dict)
            and tiles[target[1]][target[0]].get("kind") == "PASTURE"
            and "animal" not in tiles[target[1]][target[0]]
        ]
        if not targets:
            continue
        target = min(targets, key=lambda p: (_distance(pos, p), p[1], p[0]))
        reserved_targets.add(target)
        actions[idx] = ["PLACE", animal] if pos == target else _move_toward(pos, target, tiles)
        free.discard(idx)

    # Late-day liquidation is explicit.  On other turns inventories stay on
    # workers and auto-drop overnight, saving hundreds of return-path moves.
    for idx, (pos, inv) in enumerate(zip(positions, inventories)):
        if idx not in free:
            continue
        n = _count_inventory(inv)
        if n == 0:
            continue
        operational = sum(int(inv.get(k, 0)) for k in ("WHEAT", "FERTILIZER", "COW", "SHEEP")) if isinstance(inv, dict) else 0
        harvest_load = n - operational
        load_threshold = max(1, int(STRATEGY.get("drop_load_threshold", 30)))
        should_drop = (
            (day >= 29 and hour >= 12)
            or (hour >= 21 and harvest_load > 0)
            or harvest_load >= load_threshold
        )
        if should_drop:
            target = min(access, key=lambda p: (_distance(pos, p), p[1], p[0]))
            actions[idx] = ["DROP"] if pos in access else _move_toward(pos, target, tiles)
            free.discard(idx)

    # Minimum-distance matching within each priority band.  The slight tag
    # penalty keeps loaded feed/animal carriers focused on compatible work.
    for priority in sorted({t[0] for t in tasks}):
        pending = [t for t in tasks if t[0] == priority]
        while pending and free:
            candidates = []
            for unit in free:
                inv = inventories[unit]
                for j, task in enumerate(pending):
                    if not _eligible(task, inv):
                        continue
                    dist = _distance(positions[unit], task[1])
                    # Never distract a unit carrying a purchased animal with a
                    # generic nearby job; placement unlocks its daily revenue.
                    carried_animal = any(int(inv.get(a, 0)) > 0 for a in ANIMALS) if isinstance(inv, dict) else False
                    if carried_animal and task[4] != "place":
                        dist += 100
                    if STRATEGY.get("zoned_workers") and task[4] not in ("pickup_wheat", "pickup_fertilizer", "pickup_animal"):
                        if _quadrant(task[1]) != _worker_zone(unit, unlocked):
                            dist += 20
                    candidates.append((dist, unit, task[1][1], task[1][0], j))
            if not candidates:
                break
            _, unit, _, _, task_idx = min(candidates)
            task = pending.pop(task_idx)
            actions[unit] = task[2] if positions[unit] == task[1] else _move_toward(positions[unit], task[1], tiles)
            free.remove(unit)

    return actions


def _quadrant_crop_deficits(obs):
    player = int(_get(obs, "player", 0))
    farm = _get(obs, "farms", [])[player]
    private = _get(obs, "private", {}) or {}
    tiles = _get(farm, "tiles", [])
    seeds = _get(private, "seeds", {}) or {}
    day = int(_get(obs, "day", 0))
    unlocked = set(_get(farm, "unlocked_quadrants", ["NW"]) or ["NW"])
    deficits = {crop: 0 for crop in CROPS}
    for pos, crop in _crop_plan(day).items():
        x, y = pos
        if not _active_target(pos, day, unlocked) or day > _last_plant(crop):
            continue
        tile = tiles[y][x]
        if tile is None or (isinstance(tile, dict) and tile.get("kind") == "WEED"):
            deficits[crop] += 1
    for crop in deficits:
        deficits[crop] = max(0, deficits[crop] - int(seeds.get(crop, 0)))
    return deficits


def _hire_target(day):
    target = int(_style_setting("hands"))
    if STRATEGY.get("top_hire_ramp"):
        if day == 0:
            return min(4, target)
        if day <= 3:
            return min(5, target)
        if day <= 6:
            return min(6, target)
        if day <= 8:
            return min(7, target)
        if day == 9:
            return min(9, target)
        if day == 10:
            return min(10, target)
        if day <= 28:
            return target
        return min(6, target)
    if day == 0:
        return min(7, target)
    if day == 1:
        return min(4, target)
    if day == 2:
        return min(7, target)
    if day <= 4:
        return min(8, target)
    if day <= 28:
        return target
    # Final-day labour pays for itself by harvesting and liquidating the last
    # livestock output; one farmer cannot clear a mature fourteen-animal farm.
    return min(6, target)


def _hire_costs(target, already):
    fib_a, fib_b = 1, 1
    costs = []
    for index in range(max(0, int(target))):
        if index >= max(0, int(already)):
            costs.append(fib_a)
        fib_a, fib_b = fib_b, fib_a + fib_b
    return costs


def _safe_buy_price(price):
    """Budget for simultaneous opponent orders moving a market price."""
    price = max(1, int(price))
    pct = max(0, int(STRATEGY.get("price_buffer_pct", 10)))
    return max(price + 2, (price * (100 + pct) + 99) // 100)


def _observe_opponent(obs):
    """Accumulate public evidence and softly gate strategic experts.

    The features deliberately describe economic exposure rather than an
    opponent identity.  Thus an unseen agent with the same public portfolio
    receives the same response, while ambiguous portfolios remain blended
    with the broadly robust base strategy.
    """
    global _OPPONENT_STYLE, _EXPERT_EVIDENCE, _MARKET_ANIMAL_SHARE
    day = int(_get(obs, "day", 0))
    hour = int(_get(obs, "hour", 0))
    if day == 0 and hour == 0:
        _OPPONENT_STYLE = None
        _EXPERT_EVIDENCE = {}
        _MARKET_ANIMAL_SHARE = None
    market = _get(obs, "market", {}) or {}
    prices = _get(market, "prices", {}) or {}
    cow_roi = max(1.0, float(prices.get("MILK", 160))) / float(ANIMALS["COW"]["cost"])
    sheep_roi = max(1.0, float(prices.get("WOOL", 200))) / float(ANIMALS["SHEEP"]["cost"])
    sensitivity = max(0.1, float(STRATEGY.get("animal_price_sensitivity", 2.0)))
    cow_weight = cow_roi ** sensitivity
    sheep_weight = sheep_roi ** sensitivity
    _MARKET_ANIMAL_SHARE = cow_weight / (cow_weight + sheep_weight)
    farms = _get(obs, "farms", []) or []
    player = int(_get(obs, "player", 0))
    if day > 12 or len(farms) < 2:
        return
    opponent = farms[1 - player]
    tiles = [tile for row in (_get(opponent, "tiles", []) or []) for tile in row if isinstance(tile, dict)]
    plants = sum(tile.get("kind") == "PLANT" for tile in tiles)
    wheat = sum(tile.get("crop") == "WHEAT" for tile in tiles)
    strawberries = sum(tile.get("crop") == "STRAWBERRY" for tile in tiles)
    cows = sum(tile.get("animal") == "COW" for tile in tiles)
    sheep = sum(tile.get("animal") == "SHEEP" for tile in tiles)
    animals = sum(tile.get("animal") in ANIMALS for tile in tiles)
    land = len(_get(opponent, "unlocked_quadrants", []) or [])

    evidence = {}
    if day <= 3 and sheep >= 2 and cows == 0:
        evidence["EARLY_SHEEP"] = 1.0
    # Hak-like capital engine: a second quadrant full of wheat precedes a
    # large cattle wave.  Detecting it before day four prevents feed-price
    # contention from deleting our opening herd.
    if day <= 4 and land >= 2 and plants >= 28 and wheat >= 20 and animals <= 1:
        evidence["WHEAT_RUSH"] = 1.0

    # Livestock evidence is split by the opponent's exposed market demand.
    # Extreme sheep portfolios get their own counter; mixed portfolios use
    # the cow-heavy response already validated in v5.
    clear_livestock = (day <= 6 and animals >= 5 and plants <= 5) or (
        7 <= day <= 12 and animals >= 8 and plants <= 20 and strawberries <= 2
    )
    partial_livestock = 5 <= day <= 12 and animals >= 4 and plants <= 12 and strawberries <= 2
    if clear_livestock or partial_livestock:
        name = "SHEEP_RUSH" if sheep >= 5 and sheep >= 3 * max(1, cows) else "COW_RUSH"
        evidence[name] = 0.95 if clear_livestock else min(0.75, 0.35 + 0.08 * (animals - 4))

    # Repeated high-value crops imply a liquidity-sensitive strategy.  Once a
    # clear livestock regime is established we retain that earlier structural
    # signal: a late strawberry patch should not erase the proven counter.
    livestock_locked = max(
        float(_EXPERT_EVIDENCE.get("COW_RUSH", 0)),
        float(_EXPERT_EVIDENCE.get("SHEEP_RUSH", 0)),
        float(evidence.get("COW_RUSH", 0)),
        float(evidence.get("SHEEP_RUSH", 0)),
    ) >= 0.9
    if not livestock_locked:
        if (day >= 7 and strawberries >= 16) or (
            day >= 10 and strawberries >= 12 and plants >= 25
        ):
            evidence["PREMIUM_CROP"] = 1.0
        elif day >= 7 and strawberries >= 8:
            evidence["PREMIUM_CROP"] = min(0.75, 0.25 + strawberries / 40)

    for name, confidence in evidence.items():
        _EXPERT_EVIDENCE[name] = max(float(_EXPERT_EVIDENCE.get(name, 0)), confidence)
    if _EXPERT_EVIDENCE:
        _OPPONENT_STYLE = max(
            _EXPERT_EVIDENCE,
            key=lambda name: (_EXPERT_EVIDENCE[name], name == "WHEAT_RUSH", name),
        )


def _animal_purchase_cap():
    return max(0, int(_style_setting("animal_cap")))


def _market_orders(obs):
    player = int(_get(obs, "player", 0))
    farm = _get(obs, "farms", [])[player]
    private = _get(obs, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    inventories = _get(private, "inventories", []) or []
    market = _get(obs, "market", {}) or {}
    prices = _get(market, "prices", {}) or {}
    day = int(_get(obs, "day", 0))
    unlocked = list(_get(farm, "unlocked_quadrants", ["NW"]) or ["NW"])
    orders = []
    budget = float(_get(farm, "money", 0))

    # Cash conversion first funds all later orders in the same turn.
    fertilizer = int(shed.get("FERTILIZER", 0))
    fertilizer_reserve = 0
    if STRATEGY.get("fertilizer_roi") is not None and day < 29:
        fertilizer_reserve = min(fertilizer, len(_fertilizer_positions(obs)) + 3)
    fertilizer_sale = max(0, fertilizer - fertilizer_reserve)
    if fertilizer_sale > 0:
        orders.append(["SELL", "FERTILIZER", fertilizer_sale])
        budget += fertilizer_sale * float(prices.get("FERTILIZER", 1)) * 0.95
    for item in SELLABLE:
        quantity = int(shed.get(item, 0))
        if quantity > 0:
            orders.append(["SELL", item, quantity])
            budget += quantity * float(prices.get(item, 1)) * 0.95
    # Wheat is working capital for feed.  Sell only a large surplus or all of
    # it on the final day.
    counts = _asset_counts(obs)
    animal_count = sum(counts.values())
    wheat_total = int(shed.get("WHEAT", 0)) + sum(int(inv.get("WHEAT", 0)) for inv in inventories if isinstance(inv, dict))
    wheat_reserve = 0 if day >= 29 else animal_count + 3
    wheat_sale = max(0, int(shed.get("WHEAT", 0)) - wheat_reserve)
    if wheat_sale > 0:
        orders.append(["SELL", "WHEAT", wheat_sale])
        budget += wheat_sale * float(prices.get("WHEAT", 1)) * 0.95

    target_counts = {animal: 0 for animal in ANIMALS}
    unlocked_set = set(unlocked)
    for pos, animal in _animal_plan().items():
        if _animal_site_active(pos, day, unlocked_set):
            target_counts[animal] += 1

    # Maintenance comes before growth.  Secure a small critical crew first,
    # then feed, then finish the desired crew.  This ordering survives order
    # caps and prevents land/livestock purchases from consuming tomorrow's
    # operating cash.
    target_hires = _hire_target(day)
    already = int(_get(farm, "hires_today", 0))
    hire_costs = _hire_costs(target_hires, already)
    critical_target = min(target_hires, 5 if day == 0 else 2 if day <= 4 else 4)
    critical_costs = _hire_costs(critical_target, already)
    hired_costs = 0
    for cost in critical_costs:
        if len(orders) >= MAX_ORDERS or budget < cost:
            break
        orders.append(["HIRE"])
        budget -= cost
        hired_costs += 1

    # Keep one full feeding plus a small buffer.  Use a buffered unit price so
    # a simultaneous opponent purchase cannot invalidate the last hire.
    feed_days = max(1, int(STRATEGY.get("feed_days_buffer", 1)))
    desired_wheat = 0 if day >= 29 else animal_count * feed_days + 2
    feed_deficit = max(0, desired_wheat - wheat_total)
    wheat_price = _safe_buy_price(prices.get("WHEAT", 25))
    buy_feed = min(feed_deficit, int(budget // wheat_price))
    if buy_feed > 0 and len(orders) < MAX_ORDERS:
        orders.append(["BUY_PRODUCT", "WHEAT", buy_feed])
        budget -= buy_feed * wheat_price

    liquidity_floor = 0 if day >= 15 else int(STRATEGY.get("early_liquidity_floor", 0))
    for cost in hire_costs[hired_costs:]:
        if len(orders) >= MAX_ORDERS or budget - cost < liquidity_floor:
            break
        orders.append(["HIRE"])
        budget -= cost

    deficits = _quadrant_crop_deficits(obs)
    # Stage the initial plan exactly; expansion fills premium crops only while
    # their observed price still covers the remaining-yield break-even point.
    activation = {
        "MELON": 0,
        "CARROT": 0,
        "WHEAT": 0,
        "STRAWBERRY": int(STRATEGY.get("strawberry_activation_day", 4)),
        "TOMATO": 8,
    }
    operating_reserve = max(int(_style_setting("cash_reserve")), (animal_count * feed_days + 2) * wheat_price)
    for crop in ("WHEAT", "MELON", "CARROT", "STRAWBERRY", "TOMATO"):
        if day < activation[crop] or len(orders) >= MAX_ORDERS:
            continue
        if crop == "STRAWBERRY" and float(prices.get(crop, 120)) < 35:
            continue
        if crop == "MELON" and day > 10 and float(prices.get(crop, 250)) < 35:
            continue
        needed = deficits[crop]
        affordable = int(max(0, budget - operating_reserve) // CROPS[crop]["seed"])
        quantity = min(needed, affordable)
        if crop == "MELON" and day == 0 and STRATEGY.get("opening_melon_day0_cap") is not None:
            quantity = min(quantity, max(0, int(STRATEGY["opening_melon_day0_cap"])))
        elif crop == "MELON" and day <= 3 and STRATEGY.get("opening_melon_early_cap") is not None:
            quantity = min(quantity, max(0, int(STRATEGY["opening_melon_early_cap"])))
        if quantity > 0 and len(orders) < MAX_ORDERS:
            orders.append(["BUY_SEED", crop, quantity])
            budget -= quantity * CROPS[crop]["seed"]

    # Expand only with capital left after labour, feed, crops, and a cash
    # reserve.  At most two animals are added per day to avoid workload shocks.
    land_cost = 0
    if day >= int(STRATEGY.get("land_ne_day", 5)) and "NE" not in unlocked and budget - operating_reserve >= 1000:
        land_cost = 1000
    elif (
        day >= int(STRATEGY.get("land_sw_day", 10))
        and "NE" in unlocked
        and "SW" not in unlocked
        and budget - operating_reserve >= 2000
    ):
        land_cost = 2000
    if land_cost and len(orders) < MAX_ORDERS:
        orders.append(["BUY_LAND"])
        budget -= land_cost

    remaining_animal_slots = _animal_purchase_cap()
    for animal in ("COW", "SHEEP"):
        needed = max(0, target_counts[animal] - counts[animal])
        capital_per_animal = ANIMALS[animal]["cost"] + 2 * wheat_price
        affordable = int(max(0, budget - operating_reserve) // capital_per_animal)
        quantity = min(needed, affordable, remaining_animal_slots)
        if quantity > 0 and len(orders) < MAX_ORDERS:
            orders.append(["BUY_ANIMAL", animal, quantity])
            budget -= quantity * capital_per_animal
            remaining_animal_slots -= quantity

    return orders[:MAX_ORDERS]


def agent(obs):
    """Kaggle entry point."""
    try:
        if STRATEGY.get("use_fixed_schedule"):
            version = STRATEGY.get("fixed_schedule_version")
            player = int(_get(obs, "player", 0))
            use_radiant = version == "v11" and player == int(
                STRATEGY.get("v11_radiant_player", 0)
            )
            if use_radiant:
                step = min(max(0, int(_get(obs, "step", 0))), len(_V11_RADIANT_SCHEDULE) - 1)
                schedule = _v11_radiant_schedule(obs, step)
            elif version == "v18":
                board_name = _V18_RUNTIME["board_by_seat"][str(1 if player == 1 else 0)]
                schedule = _V18_RUNTIME["experts"][board_name]["actions"]
            elif version == "v17":
                schedule = _V17_SCHEDULE
            elif version == "v16":
                schedule = _v16_core_schedule(obs)
            elif version in {"v13", "v14", "v15"}:
                schedule = _V13_SENKIN_SCHEDULE
            elif version == "v12":
                schedule = _V12_SYOUYA_SCHEDULE
            elif version in {"v10", "v11"}:
                schedule = _V10_SCHEDULE
            else:
                schedule = _FIXED_SCHEDULE
            step = min(max(0, int(_get(obs, "step", 0))), len(schedule) - 1)
            action = schedule[step]
            raw = (
                _v15_senkin_action(obs, step)
                if version == "v15"
                else _v14_senkin_action(obs, step)
                if version == "v14"
                else _v13_senkin_action(obs, step)
                if version == "v13"
                else _v16_senkin_action(obs, step)
                if version == "v16"
                else _v18_closed_loop_action(obs, step)
                if version == "v18"
                else _v17_learned_action(obs, step)
                if version == "v17"
                else _v12_syouya_action(obs, step)
                if version == "v12"
                else action or {"farmer": ["PASS"], "hands": [], "market": []}
            )
            use_interference = (
                (version == "v12" and STRATEGY.get("v12_market_interference"))
                or (use_radiant and STRATEGY.get("v11_radiant_market_interference"))
                or (version not in {"v12", "v13", "v14", "v15", "v16", "v17", "v18"} and not use_radiant)
            )
            overlaid = (
                _apply_market_interference(obs, raw)
                if use_interference
                else _copy_action(raw)
            )
            return _apply_fixed_board_adaptation(obs, overlaid)
        _observe_opponent(obs)
        unit_actions = _assign_actions(obs)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:],
            "market": _market_orders(obs),
        }
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}


_PUBLIC_V18_AGENT = agent
_C17_FINAL_SELLABLE = (
    "STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT",
    "WHEAT", "FERTILIZER",
)


def agent(obs):
    """Align live hands and liquidate uncovered shed stock on the last step."""
    action = _copy_action(_PUBLIC_V18_AGENT(obs))
    player = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or []
    farm = farms[player] if 0 <= player < len(farms) else {}
    expected = len(_get(farm, "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = hands[:expected]
    step = int(_get(obs, "step", 0) or 0)
    if step == 718:
        market = list(action.get("market") or [])
        already = {}
        for order in market:
            if (
                isinstance(order, list)
                and len(order) >= 3
                and order[0] == "SELL"
            ):
                already[order[1]] = already.get(order[1], 0) + max(
                    0, int(order[2] or 0)
                )
        private = _get(obs, "private", {}) or {}
        shed = _get(private, "shed", {}) or {}
        for item in _C17_FINAL_SELLABLE:
            if len(market) >= MAX_ORDERS:
                break
            quantity = max(0, int(_get(shed, item, 0) or 0) - already.get(item, 0))
            if quantity > 0:
                market.append(["SELL", item, quantity])
        action["market"] = market
    return action


_C17_PREACTION_SHED_AGENT = agent
_C17_SHED_CAPACITY = 100
_C17_ANIMAL_STRUCTURES = {
    "GOOSE": "COOP",
    "COW": "PASTURE",
    "SHEEP": "PASTURE",
}


def _c17_project_terminal_shed(obs, action):
    """Mirror legal DROP/PLACE deposits that execute before the market."""
    player = int(_get(obs, "player", 0) or 0)
    farms = list(_get(obs, "farms", []) or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    private = _get(obs, "private", {}) or {}
    projected = dict(_get(private, "shed", {}) or {})
    inventories = list(_get(private, "inventories", []) or [])
    positions = [
        _get(farm, "farmer", None),
        *list(_get(farm, "hands", []) or []),
    ]
    actor_actions = [
        action.get("farmer", ["PASS"]),
        *list(action.get("hands") or []),
    ]
    tiles = list(_get(farm, "tiles", []) or [])
    board_size = len(tiles) or 10
    half = board_size // 2
    shed_access = {
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    }

    for actor_index, actor_action in enumerate(actor_actions):
        if actor_index >= len(positions) or actor_index >= len(inventories):
            continue
        position = positions[actor_index]
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        x, y = int(position[0]), int(position[1])
        if (x, y) not in shed_access:
            continue
        if not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
            continue
        tile = tiles[y][x]
        if tile == "LOCKED":
            continue
        if not isinstance(actor_action, list) or not actor_action:
            continue
        inventory = dict(inventories[actor_index] or {})
        op = actor_action[0]

        if op == "DROP":
            for item, raw_quantity in inventory.items():
                quantity = max(0, int(raw_quantity or 0))
                room = max(0, _C17_SHED_CAPACITY - sum(projected.values()))
                deposited = min(quantity, room)
                if deposited > 0:
                    projected[item] = projected.get(item, 0) + deposited
            continue

        if op != "PLACE" or len(actor_action) < 2:
            continue
        item = actor_action[1]
        structure = _C17_ANIMAL_STRUCTURES.get(item)
        if (
            structure is not None
            and isinstance(tile, dict)
            and tile.get("kind") == structure
            and "animal" not in tile
        ):
            continue
        try:
            requested = int(actor_action[2]) if len(actor_action) >= 3 else 1
        except (TypeError, ValueError):
            continue
        quantity = min(max(0, requested), max(0, int(inventory.get(item, 0) or 0)))
        room = max(0, _C17_SHED_CAPACITY - sum(projected.values()))
        deposited = min(quantity, room)
        if deposited > 0:
            projected[item] = projected.get(item, 0) + deposited

    return projected


def agent(obs):
    """Sell stock deposited by terminal unit actions without changing them."""
    action = _copy_action(_C17_PREACTION_SHED_AGENT(obs))
    if int(_get(obs, "step", 0) or 0) != 718:
        return action
    projected = _c17_project_terminal_shed(obs, action)
    market = list(action.get("market") or [])
    already = {}
    for order in market:
        if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL":
            already[order[1]] = already.get(order[1], 0) + max(
                0, int(order[2] or 0)
            )
    for item in _C17_FINAL_SELLABLE:
        if len(market) >= MAX_ORDERS:
            break
        quantity = max(0, int(projected.get(item, 0) or 0) - already.get(item, 0))
        if quantity > 0:
            market.append(["SELL", item, quantity])
    action["market"] = market
    return action


def _kaggle_submission_entrypoint(obs):
    """Keep the final callable selected by Kaggle bound to the public agent."""
    return agent(obs)
