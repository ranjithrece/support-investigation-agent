# Demo output (scripts/demo.py --scenario all, qwen3:8b, local)

Captured from a real run; all 4 paths passed.

```text
───────────────────────────── 1. Normal resolution ─────────────────────────────
AGENT ASKS: Could you please provide the router model and firmware version that 
were updated last week?
USER ANSWERS: All offices are affected; the router firmware was updated last 
week.
╭─────────────────────────────────── Result ───────────────────────────────────╮
│ Case            C-20261118-772A35                                            │
│ Classification  Technical Support / Incident / medium  tags=[]               │
│ Confidence      0.84                                                         │
│ Escalation      no                                                           │
│ Grounded        True (score 1.0)                                             │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────── Analysis ──────────────────────────────────╮
│ The frequent VPN disconnections are likely caused by the recent router       │
│ firmware update, which may have introduced a compatibility issue or bug.     │
│ Multiple cases (C-20261118-124551, C-20261118-B9398E, H-000411, H-000273,    │
│ H-000008) suggest that providing the router model and firmware version is    │
│ critical for diagnosis. Some cases (H-000273, H-000008) also recommend       │
│ checking for firmware updates or rolling back to a previous version. While   │
│ H-021959 suggests network congestion as an alternative cause, the majority   │
│ of cases point to the firmware update as the primary issue.                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────── Resolution ─────────────────────────────────╮
│ We understand the frustration of frequent VPN disconnections impacting       │
│ remote access. Please provide the router model, firmware version, and any    │
│ recent network changes. These details will help us identify the root cause   │
│ and suggest solutions. If the issue persists after this, please reach out    │
│ for further assistance.                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
                                    Evidence                                    
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ ticket            ┃ src  ┃ queue/type           ┃ sim  ┃ summary             ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ C-20261118-124551 │ live │ Technical            │ 0.96 │ Frequent VPN        │
│                   │      │ Support/Incident     │      │ disconnections: Our │
│                   │      │                      │      │ VPN keeps           │
│                   │      │                      │      │ disconnecting every │
│                   │      │                      │      │ few min             │
│ C-20261118-B9398E │ live │ Technical            │ 0.89 │ Frequent VPN        │
│                   │      │ Support/Incident     │      │ disconnections: Our │
│                   │      │                      │      │ VPN keeps           │
│                   │      │                      │      │ disconnecting every │
│                   │      │                      │      │ few min             │
│ H-000411          │ hist │ Technical            │ 0.80 │ Connectivity        │
│                   │      │ Support/Incident     │      │ Fluctuations: Dear  │
│                   │      │                      │      │ Support Team,  We   │
│                   │      │                      │      │ are facing sporadic │
│ H-000273          │ hist │ Product              │ 0.80 │ VPN Connection      │
│                   │      │ Support/Incident     │      │ Issue: Dear Support │
│                   │      │                      │      │ Team,  We are       │
│                   │      │                      │      │ encountering a      │
│                   │      │                      │      │ failur              │
│ H-000008          │ hist │ Product              │ 0.79 │ VPN Access Issue:   │
│                   │      │ Support/Incident     │      │ Customer Support,   │
│                   │      │                      │      │ We are encountering │
│                   │      │                      │      │ a disruption        │
│ H-021959          │ hist │ Product              │ 0.79 │ Frequent Failures   │
│                   │      │ Support/Incident     │      │ in VPN Router       │
│                   │      │                      │      │ Connection: Random  │
│                   │      │                      │      │ disconnections are  │
└───────────────────┴──────┴──────────────────────┴──────┴─────────────────────┘
tools chosen by the model: ['search_similar_tickets', 'get_ticket_by_id', 
'get_queue_tag_statistics']
indexed for future retrieval: True
──── 2. Clarification + resume (interrupt survives a fresh graph instance) ─────
AGENT ASKS (clarification): Could you please specify which product or service is
not working and any recent changes you made to your system or network?
USER ANSWERS: It is our company VPN client; it disconnects every few minutes on 
my laptop.
╭─────────────────────────────────── Result ───────────────────────────────────╮
│ Case            C-20261118-36DCE3                                            │
│ Classification  Product Support / Incident / medium  tags=[]                 │
│ Confidence      0.82                                                         │
│ Escalation      no                                                           │
│ Grounded        True (score 1.0)                                             │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────── Analysis ──────────────────────────────────╮
│ The customer is experiencing frequent disconnections with their company's    │
│ VPN client. Multiple historical cases (C-20261118-070F0F, C-20261118-FE1515, │
│ C-20261118-124551, C-20261118-772A35, C-20261118-B9398E) indicate that       │
│ providing router model, firmware version, and recent network changes is a    │
│ common step to identify the root cause. The case H-007775 suggests gathering │
│ error messages and steps taken. These steps are consistent across the        │
│ evidence.                                                                    │
╰──────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────── Resolution ─────────────────────────────────╮
│ We understand the frustration of frequent VPN disconnections impacting your  │
│ work. Please provide the router model, firmware version, and any recent      │
│ network changes. These details will help us identify the root cause and      │
│ suggest solutions. If the issue persists after this, please reach out for    │
│ further assistance.                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
                                    Evidence                                    
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ ticket            ┃ src  ┃ queue/type           ┃ sim  ┃ summary             ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ C-20261118-070F0F │ live │ Product              │ 0.85 │ It is not working:  │
│                   │      │ Support/Incident     │      │ Hi, it stopped      │
│                   │      │                      │      │ working. Please     │
│                   │      │                      │      │ help.               │
│ C-20261118-FE1515 │ live │ Product              │ 0.85 │ It is not working:  │
│                   │      │ Support/Incident     │      │ Hi, it stopped      │
│                   │      │                      │      │ working. Please     │
│                   │      │                      │      │ help.               │
│ C-20261118-124551 │ live │ Technical            │ 0.80 │ Frequent VPN        │
│                   │      │ Support/Incident     │      │ disconnections: Our │
│                   │      │                      │      │ VPN keeps           │
│                   │      │                      │      │ disconnecting every │
│                   │      │                      │      │ few min             │
│ C-20261118-772A35 │ live │ Technical            │ 0.80 │ Frequent VPN        │
│                   │      │ Support/Incident     │      │ disconnections: Our │
│                   │      │                      │      │ VPN keeps           │
│                   │      │                      │      │ disconnecting every │
│                   │      │                      │      │ few min             │
└───────────────────┴──────┴──────────────────────┴──────┴─────────────────────┘
clarifications recorded: [{'question': 'Could you please specify which product 
or service is not working and any recent changes you made to your system or 
network?', 'answer': 'It is our company VPN client; it disconnects every few 
minutes on my laptop.'}]
─────────── 3. Long-term memory reuse across separate conversations ────────────
AGENT ASKS: Could you please confirm if the file server connection issue started
after any recent network changes, firmware updates, or new device connections?
USER ANSWERS: Cisco AnyConnect on macOS laptops, all users are affected.
memories stored after ticket 1:
[
    {
        'kind': 'environment',
        'text': 'Customer uses Cisco AnyConnect VPN on macOS laptops.'
    },
    {
        'kind': 'pattern',
        'text': 'Users lose file server connection when using Cisco AnyConnect 
VPN.'
    },
    {
        'kind': 'outcome',
        'text': 'Past accepted case: Cannot reach file server over VPN (queue 
Technical Support)'
    }
]
ticket 2 (new thread) intake loaded: {'memory_items': 3, 'history': 1}
╭─────────────────────────────────── Result ───────────────────────────────────╮
│ Case            C-20261118-7F14D1                                            │
│ Classification  Technical Support / Incident / medium  tags=[]               │
│ Confidence      0.98                                                         │
│ Escalation      no                                                           │
│ Grounded        True (score 1.0)                                             │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────── Analysis ──────────────────────────────────╮
│ The issue of frequent VPN disconnections impacting remote access to file     │
│ servers is well-documented in multiple historical tickets. The common thread │
│ across these cases is the need for detailed network information to diagnose  │
│ the root cause. The most common path involves gathering router model,        │
│ firmware version, and recent network changes. While some tickets mention     │
│ specific devices like Cisco AnyConnect, others focus on general router       │
│ configurations. Escalation is not required as the steps provided are         │
│ sufficient for further troubleshooting.                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────── Resolution ─────────────────────────────────╮
│ We understand the frustration of frequent VPN disconnections impacting       │
│ remote access. Please provide the router model, firmware version, and any    │
│ recent network changes. These details will help us identify the root cause   │
│ and suggest solutions. If the issue persists after this, please reach out    │
│ for further assistance.                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
                                    Evidence                                    
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ ticket            ┃ src  ┃ queue/type           ┃ sim  ┃ summary             ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ C-20261118-2F300A │ live │ Technical            │ 0.86 │ VPN problem again:  │
│                   │      │ Support/Incident     │      │ VPN dropped again   │
│                   │      │                      │      │ for another user    │
│                   │      │                      │      │ today.              │
│ C-20261118-5A05E8 │ live │ Technical            │ 0.86 │ VPN problem again:  │
│                   │      │ Support/Incident     │      │ VPN dropped again   │
│                   │      │                      │      │ for another user    │
│                   │      │                      │      │ today.              │
│ C-20261118-124551 │ live │ Technical            │ 0.71 │ Frequent VPN        │
│                   │      │ Support/Incident     │      │ disconnections: Our │
│                   │      │                      │      │ VPN keeps           │
│                   │      │                      │      │ disconnecting every │
│                   │      │                      │      │ few min             │
│ C-20261118-772A35 │ live │ Technical            │ 0.71 │ Frequent VPN        │
│                   │      │ Support/Incident     │      │ disconnections: Our │
│                   │      │                      │      │ VPN keeps           │
│                   │      │                      │      │ disconnecting every │
│                   │      │                      │      │ few min             │
│ C-20261118-B9398E │ live │ Technical            │ 0.69 │ Frequent VPN        │
│                   │      │ Support/Incident     │      │ disconnections: Our │
│                   │      │                      │      │ VPN keeps           │
│                   │      │                      │      │ disconnecting every │
│                   │      │                      │      │ few min             │
│ H-000273          │ hist │ Product              │ 0.69 │ VPN Connection      │
│                   │      │ Support/Incident     │      │ Issue: Dear Support │
│                   │      │                      │      │ Team,  We are       │
│                   │      │                      │      │ encountering a      │
│                   │      │                      │      │ failur              │
└───────────────────┴──────┴──────────────────────┴──────┴─────────────────────┘
VPN problem again <- remembered history
────────────────── 4. Newly resolved case becomes searchable ───────────────────
[
    ('C-20261118-124551', 'live', 0.827),
    ('C-20261118-B9398E', 'live', 0.824),
    ('C-20261118-772A35', 'live', 0.827),
    ('H-021959', 'hist', 0.696),
    ('H-020717', 'hist', 0.693),
    ('H-000411', 'hist', 0.714),
    ('C-20261118-7191AC', 'live', 0.652),
    ('C-20261118-E1A49D', 'live', 0.652)
]
OK: live case(s) in results: ['C-20261118-124551', 'C-20261118-B9398E', 
'C-20261118-772A35', 'C-20261118-7191AC', 'C-20261118-E1A49D']
new ticket cited live (agent-resolved) cases as evidence: ['C-20261118-124551', 
'C-20261118-772A35', 'C-20261118-B9398E']

```
