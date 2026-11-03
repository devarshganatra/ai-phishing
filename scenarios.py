"""Scenario registry for the Intent-vs-Register matched-pair study.

Each scenario fixes a topic, a sender role and a set of surface cues (a deadline, a link,
an authority figure). For every scenario we build three request variants that hold those
surface cues constant while changing only the sender's *intent*:

  P       the phishing request  (reused verbatim from the existing repo, label 1)
  B_hard  a genuinely legitimate email on the SAME topic with the SAME cues (label 0)
  B_easy  a plain legitimate email on the same topic, no urgency/link/authority cues (label 0)

The phishing prompts are imported from generate_ai_emails.py so this file adds no new
phishing-authoring instructions; it only pairs each with its legitimate twins.
"""
from generate_ai_emails import PROMPT_TEMPLATES as PHISHING_PROMPTS

# One legitimate-twin description per scenario index, matched to PHISHING_PROMPTS above.
# B_hard keeps the same realistic trigger (password expiry, a real invoice, a real delivery)
# so that lexical cues alone cannot separate it from the phishing version; the difference is
# that it is a real, benign message pointing to the official channel.
_LEGIT_TWINS = [
    "a real IT helpdesk notice that the employee's password will expire in 3 days, telling them to "
    "change it through the company's normal internal password portal at the usual intranet address",
    "a genuine email from a CEO to Finance approving a already-contracted, previously-discussed vendor "
    "payment through the company's standard procurement system",
    "a genuine PayPal service email confirming a routine login from a recognised device and telling the "
    "user no action is needed, with a link to the real paypal.com help centre",
    "a genuine HR email announcing an updated company policy, attached as an internal document and also "
    "available on the company intranet",
    "a genuine renewal receipt from a software company for a subscription the user actually bought, with "
    "the company's real billing-portal address for managing the plan",
    "a genuine delivery update from a courier for a parcel the user is expecting, with the real tracking "
    "number and the courier's official tracking page",
    "a genuine notice from a company's rewards programme that the user has points to redeem, directing "
    "them to log in through the official account page they normally use",
    "a genuine bank security email confirming the user successfully enabled two-factor authentication, "
    "pointing to the bank's official app and help centre",
    "a genuine Microsoft 365 email reminding the user that routine maintenance is scheduled, with the "
    "official admin-centre address and no action required",
    "a genuine email from a colleague sharing a real Q3 report through the company's normal document "
    "system, referring to a meeting the team actually had",
]

# A plainer, cue-free legitimate email on the same broad topic (no deadline, link or authority push).
_EASY_TWINS = [
    "a short friendly note from a coworker asking if the recipient wants to grab coffee this week",
    "a brief internal update from a manager thanking the finance team for a smooth quarter close",
    "a short message from a friend sharing holiday photos and asking how the recipient has been",
    "a short HR email welcoming a new team member and listing their start date",
    "a brief thank-you email from a small business to a long-time customer, no offer or link",
    "a short message from a neighbour about a community gardening weekend",
    "a brief newsletter-style note recapping a local book club's last meeting",
    "a short message from a relative asking about weekend plans",
    "a brief IT note letting staff know the office wifi name changed, for information only",
    "a short message from a colleague sharing notes from a meeting the team had",
]

assert len(PHISHING_PROMPTS) == len(_LEGIT_TWINS) == len(_EASY_TWINS), "scenario lists must align"

NUM_SCENARIOS = len(PHISHING_PROMPTS)

LABELS = {'P': 1, 'B_hard': 0, 'B_easy': 0, 'P_R': 1, 'B_hard_R': 0}

# One system prompt for every condition: the only thing that changes between twins is the request.
SYSTEM_PROMPT = (
    "You are helping build a labeled email dataset for training and evaluating phishing detectors. "
    "Write exactly the email that is requested. Output only the email: a 'Subject:' line, a blank line, "
    "then the body. No commentary, no notes, no markdown."
)

# Appended to every request so twins differ only in intent, not in having unfilled template slots.
STYLE_RULES = (
    " Use concrete, fictional details: names, company names, URLs, phone numbers, amounts and dates. "
    "Never use square-bracket or curly-brace placeholders such as [Name] or [Link]. "
    "Do not add security disclaimers, warnings about phishing, or notes about the email being an example."
)

# Register targets for the rewrite stage: same intent and facts, different writing style only.
REGISTER_REWRITES = {
    'terse_internal': (
        "Rewrite the email below in a terse internal-business register: short sentences, no pleasantries, "
        "as if sent quickly between colleagues. Keep every fact, name, link, number and the sender's "
        "intent exactly the same. Change only the wording and tone. Output only the rewritten email."
    ),
    'polished_cs': (
        "Rewrite the email below in a polished, warm customer-service register: complete sentences, "
        "courteous framing. Keep every fact, name, link, number and the sender's intent exactly the "
        "same. Change only the wording and tone. Output only the rewritten email."
    ),
}


def build_request(scenario_idx, condition):
    """Return the user-prompt string for a given scenario and condition.

    P reuses the repo's phishing prompt verbatim; B_hard / B_easy request a legitimate email.
    """
    if condition == 'P':
        return PHISHING_PROMPTS[scenario_idx]
    if condition == 'B_hard':
        return "Write " + _LEGIT_TWINS[scenario_idx] + "."
    if condition == 'B_easy':
        return "Write " + _EASY_TWINS[scenario_idx] + "."
    raise ValueError(f"build_request handles base conditions only, got {condition!r}")
