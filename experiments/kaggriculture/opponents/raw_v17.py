"""Raw replay of the public v18/V17 expert schedule (Apache-2.0, Kaito Fukami
public notebook lineage). Engine no-ops invalid actions; hands clipped."""
import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    'sey_src', os.path.join(os.path.dirname(__file__), 'sey_v7.py'))
_sey = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sey)
_SCHED = _sey._V17_SCHEDULE


def agent(obs):
    step = obs.get('step', 0)
    if step >= len(_SCHED):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    act = dict(_SCHED[step])
    n_hands = len(obs.get('farms', [{}, {}])[obs.get('player', 0)].get('hands', []))
    hands = list(act.get('hands', []))[:n_hands]
    hands += [["PASS"]] * (n_hands - len(hands))
    return {"farmer": act.get('farmer', ["PASS"]), "hands": hands,
            "market": act.get('market', [])}
