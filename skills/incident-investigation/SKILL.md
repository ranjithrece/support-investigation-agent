---
name: incident-investigation
description: Investigate a ticket by forming hypotheses, retrieving targeted evidence, and comparing resolutions across several historical cases.
when_to_use: Whenever a technical, billing or service problem needs a resolution grounded in past cases.
---
# Incident investigation
1. State a short hypothesis of the root cause or needed action.
2. Use `search_similar_tickets` with a query that targets the hypothesis; add queue/type filters when the triage is confident. Try a second, differently-worded query if results are weak.
3. Read the *answers* of at least 3 cases. Note which steps recur (consensus) and which differ (conflict).
4. Use `get_ticket_by_id` to read a promising case in full. Use `get_queue_tag_statistics` to see what is typical for this queue/tag.
5. Use `get_customer_history` when the customer may have raised this before.
6. Stop when you can name consensus steps backed by 2+ cases, or when evidence is clearly insufficient or conflicting. Never copy one answer blindly; synthesize and cite ticket IDs.
