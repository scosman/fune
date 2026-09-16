/**
 * Warning copy for eval sets that contain stored multi-turn conversations.
 * The eval runner scores those items from their saved conversation, so every
 * run configuration receives identical scores for them. This depends only on
 * the item set — not on the eval's data type — so the gate takes just the
 * count reported by the score summary. Returns null when there is nothing to
 * warn about (no summary loaded yet, or no stored conversations in the set).
 */
export function multiTurnStoredScoreWarning(
  multi_turn_item_count: number | null | undefined,
): string | null {
  if (!multi_turn_item_count || multi_turn_item_count <= 0) {
    return null
  }
  const scoring =
    multi_turn_item_count === 1
      ? "1 stored multi-turn conversation. It is scored using its saved messages, so all run configurations receive an identical score for it."
      : `${multi_turn_item_count} stored multi-turn conversations. These are scored using their saved messages, so all run configurations receive identical scores for them.`
  return `This eval's dataset contains ${scoring} To compare run configurations on fresh conversations, create the eval with the eval builder.`
}
