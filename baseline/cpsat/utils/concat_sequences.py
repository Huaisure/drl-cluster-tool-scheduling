from typing import Any, Dict, List, Optional, Tuple

def assign_good_ids_by_occurrence(
	actions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

	indexed_actions: List[Tuple[int, Dict[str, Any]]] = list(enumerate(actions or []))
	indexed_actions.sort(key=lambda item: (float(item[1].get("StartTime", 0.0)), item[0]))

	occurrence_by_key: Dict[Tuple[Any, ...], int] = {}
	pr_index_by_pr: Dict[str, int] = {}
	next_pr_index = 1
	annotated: List[Dict[str, Any]] = [dict(a) for a in actions or []]

	for original_index, action in indexed_actions:
		pr_key = str(action.get("pr_id"))
		if pr_key not in pr_index_by_pr:
			pr_index_by_pr[pr_key] = next_pr_index
			next_pr_index += 1

		key = (
			action.get("MoveType"),
			action.get("pr_id"),
			action.get("step"),
		)
		occurrence = occurrence_by_key.get(key, 0) + 1
		occurrence_by_key[key] = occurrence

		annotated[original_index]["good_id"] = occurrence

	for move_id, action in enumerate(
		sorted(
			annotated,
			key=lambda item: (
				float(item.get("StartTime", 0.0)),
				float(item.get("EndTime", 0.0)),
				str(item.get("good_id")),
			),
		)
	):
		action["Move_ID"] = move_id

	return annotated
