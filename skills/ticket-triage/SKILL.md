---
name: ticket-triage
description: Classify a ticket into queue, type, priority and tags using the ticket text and historical evidence.
when_to_use: First step for every new ticket, or when the classification looks wrong.
---
# Ticket triage
1. Pick the **queue** whose past tickets most resemble this one. Prefer the queue that dominates the retrieved evidence; use the statistics tool when evidence is split.
2. **Type**: Incident = something broke or is down; Problem = recurring/root-cause issue; Request = asks for a service/info/change of access; Change = planned modification.
3. **Priority**: critical/high for outages, data loss, security, payments blocked, or many users affected; medium for degraded service with a workaround; low for questions and cosmetic issues.
4. **Tags**: up to 6 short topical words reused from similar historical tickets when sensible.
5. List facts a support agent would need but the ticket does not give (affected product, error text, scope, time started).
Do not invent details. If the ticket is too vague to classify, say so and list missing information.
