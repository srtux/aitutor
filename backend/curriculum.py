"""Curriculum definitions and deterministic Socratic response generator.

Provides multi-step curriculum problems for Linear Equations, Python Loops,
and Fraction Word Problems, complete with visual whiteboard states, hints,
binary choices, and warm, conversational affective pedagogical responses.
"""

from typing import Any

CURRICULUM_DATA: dict[str, dict[str, Any]] = {
    "linear_equations": {
        "id": "linear_equations",
        "label": "1. Linear Equations & Balancing",
        "subject": "Algebra",
        "opening_greeting": (
            "Hey! Welcome to equation balancing. Think of the equals sign like a balance scale. "
            "Let's tackle 3x + 7 = 22 together!"
        ),
        "steps": [
            {
                "step_idx": 0,
                "problem": "Solve for x: 3x + 7 = 22",
                "whiteboard_title": "Equation Balancing: 3x + 7 = 22",
                "steps": [
                    {
                        "label": "Isolate variable term by subtracting 7 from both sides",
                        "expr": "3x = 15",
                        "status": "current",
                    },
                    {
                        "label": "Solve for x by dividing both sides by 3",
                        "expr": "x = 5",
                        "status": "pending",
                    },
                ],
                "visual_diagram": "balance_scale",
                "binary_choices": [
                    "Subtract 7 from both sides",
                    "Divide both sides by 3 first",
                ],
                "socratic_hint": (
                    "To keep our scale balanced, what operation undoes adding 7 to the left side?"
                ),
                "proactive_prompt": (
                    "I notice this step looks a bit sticky. Want to break down the left side of "
                    "the equation together?"
                ),
                "expected_keywords": [
                    "subtract",
                    "minus",
                    "7",
                    "seven",
                    "15",
                    "fifteen",
                    "3x=15",
                    "3x = 15",
                    "take away 7",
                    "take away seven",
                    "x=5",
                    "x = 5",
                    "5",
                    "five",
                    "choice a",
                    "option a",
                    "subtract 7 from both sides",
                ],
            },
            {
                "step_idx": 1,
                "problem": "Solve for x: 2(x - 4) + 5 = 15",
                "whiteboard_title": "Multi-Step Equation: 2(x - 4) + 5 = 15",
                "steps": [
                    {
                        "label": "Distribute 2 into the parentheses",
                        "expr": "2x - 8 + 5 = 15",
                        "status": "current",
                    },
                    {
                        "label": "Combine constant terms on the left",
                        "expr": "2x - 3 = 15",
                        "status": "pending",
                    },
                    {
                        "label": "Add 3 to both sides",
                        "expr": "2x = 18",
                        "status": "pending",
                    },
                    {
                        "label": "Divide both sides by 2",
                        "expr": "x = 9",
                        "status": "pending",
                    },
                ],
                "visual_diagram": "balance_scale",
                "binary_choices": [
                    "Distribute 2 to get 2x - 8",
                    "Add 5 to 4 inside parentheses",
                ],
                "socratic_hint": (
                    "Before balancing, how can we expand 2(x - 4) using the distributive property?"
                ),
                "proactive_prompt": (
                    "Parentheses can be tricky. Would expanding 2 times (x - 4) help simplify things?"
                ),
                "expected_keywords": [
                    "distribute",
                    "distribute 2",
                    "2x - 8",
                    "2x-8",
                    "expand",
                    "multiply by 2",
                    "2x - 3",
                    "9",
                    "nine",
                    "x=9",
                    "x = 9",
                    "choice a",
                    "option a",
                    "distribute 2 to get 2x - 8",
                ],
            },
            {
                "step_idx": 2,
                "problem": "Solve for x: 5x - 3 = 2x + 12",
                "whiteboard_title": "Variables on Both Sides: 5x - 3 = 2x + 12",
                "steps": [
                    {
                        "label": "Collect variable terms on left by subtracting 2x",
                        "expr": "3x - 3 = 12",
                        "status": "current",
                    },
                    {
                        "label": "Add 3 to both sides",
                        "expr": "3x = 15",
                        "status": "pending",
                    },
                    {
                        "label": "Divide both sides by 3",
                        "expr": "x = 5",
                        "status": "pending",
                    },
                ],
                "visual_diagram": "balance_scale",
                "binary_choices": [
                    "Subtract 2x from both sides",
                    "Add 3 to both sides first",
                ],
                "socratic_hint": (
                    "We have x on both sides of the balance. What happens if we remove 2x from each side?"
                ),
                "proactive_prompt": (
                    "There are variables on both sides of our scale. How might we gather all the x terms together?"
                ),
                "expected_keywords": [
                    "subtract 2x",
                    "minus 2x",
                    "-2x",
                    "take away 2x",
                    "3x - 3",
                    "3x = 15",
                    "5",
                    "five",
                    "x=5",
                    "x = 5",
                    "choice a",
                    "option a",
                    "subtract 2x from both sides",
                ],
            },
        ],
    },
    "python_loops": {
        "id": "python_loops",
        "label": "2. Python Loops & Off-by-One Bugs",
        "subject": "Computer Science",
        "opening_greeting": (
            "Welcome to loop debugging! We'll track variables step by step to catch those "
            "sneaky off-by-one bugs. Let's trace our first loop!"
        ),
        "steps": [
            {
                "step_idx": 0,
                "problem": "Debug: total = 0; for i in range(1, 5): total += i (Goal: sum 1..5 inclusive)",
                "whiteboard_title": "Loop Bounds: range(1, 5) vs range(1, 6)",
                "steps": [
                    {
                        "label": "Identify values generated by range(1, 5)",
                        "expr": "i generates [1, 2, 3, 4] (sum = 10)",
                        "status": "current",
                    },
                    {
                        "label": "Adjust stop parameter to include 5",
                        "expr": "for i in range(1, 6): (sum = 15)",
                        "status": "pending",
                    },
                    {
                        "label": "Verify final total equals 15",
                        "expr": "total == 15",
                        "status": "pending",
                    },
                ],
                "visual_diagram": "loop_trace",
                "binary_choices": [
                    "Change range(1, 5) to range(1, 6)",
                    "Change total += i to total += i + 1",
                ],
                "socratic_hint": (
                    "Remember that range(start, stop) stops right before the stop value. What is the last number generated?"
                ),
                "proactive_prompt": (
                    "Loops and ranges often hide off-by-one catches. What numbers does range(1, 5) actually produce?"
                ),
                "expected_keywords": [
                    "6",
                    "six",
                    "range(1, 6)",
                    "range(1,6)",
                    "range 1 to 6",
                    "1 to 6",
                    "stop is exclusive",
                    "stops before 5",
                    "4",
                    "four",
                    "15",
                    "fifteen",
                    "choice a",
                    "option a",
                    "change range(1, 5) to range(1, 6)",
                ],
            },
            {
                "step_idx": 1,
                "problem": "Fix IndexError: scores = [92, 85, 99]; last = scores[len(scores)]",
                "whiteboard_title": "Zero-Indexed Lists & Index Bounds",
                "steps": [
                    {
                        "label": "Examine list length and valid index range",
                        "expr": "len(scores) == 3, valid indices: 0, 1, 2",
                        "status": "current",
                    },
                    {
                        "label": "Access last element without out-of-bounds error",
                        "expr": "scores[len(scores) - 1] or scores[-1]",
                        "status": "pending",
                    },
                ],
                "visual_diagram": "loop_trace",
                "binary_choices": [
                    "Use scores[len(scores) - 1] or scores[-1]",
                    "Change scores array to be 1-indexed",
                ],
                "socratic_hint": (
                    "Python lists are 0-indexed. If a list has 3 items, what are the indices of those items?"
                ),
                "proactive_prompt": (
                    "Index errors usually mean we stepped past the end. What is the index of the very first element?"
                ),
                "expected_keywords": [
                    "-1",
                    "negative 1",
                    "negative one",
                    "len - 1",
                    "len(scores) - 1",
                    "length minus 1",
                    "0, 1, 2",
                    "0-indexed",
                    "zero indexed",
                    "index 2",
                    "scores[-1]",
                    "choice a",
                    "option a",
                    "use scores[len(scores) - 1] or scores[-1]",
                ],
            },
            {
                "step_idx": 2,
                "problem": "Debug while loop: count = 0; while count <= n: count += 1 (Runs n+1 times)",
                "whiteboard_title": "While Loop Conditions: <= vs <",
                "steps": [
                    {
                        "label": "Trace iterations when n = 3",
                        "expr": "count takes 0, 1, 2, 3 (4 iterations)",
                        "status": "current",
                    },
                    {
                        "label": "Correct condition so it runs exactly n times",
                        "expr": "while count < n:",
                        "status": "pending",
                    },
                ],
                "visual_diagram": "loop_trace",
                "binary_choices": [
                    "Change while count <= n to while count < n",
                    "Start count at 1 with count <= n",
                ],
                "socratic_hint": (
                    "If count starts at 0 and checks count <= 3, how many numbers did it visit?"
                ),
                "proactive_prompt": (
                    "Let's trace count starting at 0. If n is 3, count takes values 0, 1, 2, and 3. How many times is that?"
                ),
                "expected_keywords": [
                    "<",
                    "less than",
                    "count < n",
                    "4 times",
                    "four times",
                    "n+1",
                    "strictly less",
                    "choice a",
                    "option a",
                    "change while count <= n to while count < n",
                ],
            },
        ],
    },
    "fraction_word_problems": {
        "id": "fraction_word_problems",
        "label": "3. Fraction Word Problems",
        "subject": "Arithmetic",
        "opening_greeting": (
            "Hey there! We'll use visual fraction bars to make word problems super intuitive. "
            "Let's scale this recipe together!"
        ),
        "steps": [
            {
                "step_idx": 0,
                "problem": "A recipe calls for 3/4 cup flour. We are making 2/3 of a batch. How much flour?",
                "whiteboard_title": "Recipe Scaling: 3/4 cup * 2/3 batch",
                "steps": [
                    {
                        "label": "Multiply numerators and denominators",
                        "expr": "(3 * 2) / (4 * 3) = 6/12",
                        "status": "current",
                    },
                    {
                        "label": "Simplify fraction to lowest terms",
                        "expr": "6/12 = 1/2 cup flour",
                        "status": "pending",
                    },
                ],
                "visual_diagram": "fraction_bars",
                "binary_choices": [
                    "Multiply: (3/4) * (2/3)",
                    "Find common denominator to add: 3/4 + 2/3",
                ],
                "socratic_hint": (
                    "When taking two thirds OF three fourths, what mathematical operation represents of?"
                ),
                "proactive_prompt": (
                    "Taking a fraction of another fraction can be visualized with area. Would multiplying numerator by numerator help?"
                ),
                "expected_keywords": [
                    "multiply",
                    "multiplication",
                    "times",
                    "6/12",
                    "six twelfths",
                    "1/2",
                    "one half",
                    "half",
                    "0.5",
                    "choice a",
                    "option a",
                    "multiply: (3/4) * (2/3)",
                ],
            },
            {
                "step_idx": 1,
                "problem": "Leo eats 1/3 of a pizza and Maya eats 2/5. What fraction of the pizza is left?",
                "whiteboard_title": "Pizza Sharing: 1 - (1/3 + 2/5)",
                "steps": [
                    {
                        "label": "Find common denominator for 1/3 and 2/5 (LCD = 15)",
                        "expr": "5/15 + 6/15 = 11/15 eaten",
                        "status": "current",
                    },
                    {
                        "label": "Subtract total eaten from the whole pizza",
                        "expr": "15/15 - 11/15 = 4/15 left",
                        "status": "pending",
                    },
                ],
                "visual_diagram": "fraction_bars",
                "binary_choices": [
                    "Find common denominator 15",
                    "Add numerators directly: 1+2 / 3+5",
                ],
                "socratic_hint": (
                    "To combine slices of different sizes (thirds and fifths), what common slice size can we cut them into?"
                ),
                "proactive_prompt": (
                    "We can't add thirds and fifths directly. What number do both 3 and 5 divide into evenly?"
                ),
                "expected_keywords": [
                    "15",
                    "fifteen",
                    "common denominator",
                    "11/15",
                    "eleven fifteenths",
                    "4/15",
                    "four fifteenths",
                    "lcd",
                    "lcm",
                    "choice a",
                    "option a",
                    "find common denominator 15",
                ],
            },
            {
                "step_idx": 2,
                "problem": "A trail is 7/8 mile long. Sam has hiked 5/6 mile. How much further to the end?",
                "whiteboard_title": "Trail Distance: 7/8 mile - 5/6 mile",
                "steps": [
                    {
                        "label": "Find common denominator for 8 and 6 (LCD = 24)",
                        "expr": "7/8 = 21/24 and 5/6 = 20/24",
                        "status": "current",
                    },
                    {
                        "label": "Subtract distance hiked from total trail length",
                        "expr": "21/24 - 20/24 = 1/24 mile remaining",
                        "status": "pending",
                    },
                ],
                "visual_diagram": "fraction_bars",
                "binary_choices": [
                    "Convert to denominator 24 and subtract",
                    "Subtract numerators: 7 - 5 = 2",
                ],
                "socratic_hint": (
                    "What is the least common multiple of 8 and 6 to compare the trail distances?"
                ),
                "proactive_prompt": (
                    "We need to find the difference between 7/8 and 5/6. What common denominator can we use?"
                ),
                "expected_keywords": [
                    "24",
                    "twenty four",
                    "twenty-four",
                    "common denominator",
                    "21/24",
                    "twenty-one twenty-fourths",
                    "20/24",
                    "twenty twenty-fourths",
                    "1/24",
                    "one twenty-fourth",
                    "choice a",
                    "option a",
                    "convert to denominator 24 and subtract",
                ],
            },
        ],
    },
}


def list_topics() -> list[dict[str, Any]]:
    """Return all available curriculum topics for the topic selector."""
    results = []
    for topic_id, data in CURRICULUM_DATA.items():
        results.append(
            {
                "id": topic_id,
                "label": data["label"],
                "subject": data["subject"],
                "initial_whiteboard": get_initial_whiteboard(topic_id, step_idx=0),
            }
        )
    return results


def get_topic(topic_id: str) -> dict[str, Any]:
    """Retrieve curriculum data for a topic by ID with fallback to linear_equations."""
    return CURRICULUM_DATA.get(topic_id, CURRICULUM_DATA["linear_equations"])


def get_opening_greeting(topic_id: str) -> str:
    """Return the 1-2 sentence welcoming greeting for the topic."""
    topic = get_topic(topic_id)
    return topic.get(
        "opening_greeting",
        "Welcome! Let's work through this problem step by step together.",
    )


def get_proactive_prompt(topic_id: str, step_idx: int = 0) -> str:
    """Return the proactive silence intervention prompt for a given step."""
    topic = get_topic(topic_id)
    steps = topic.get("steps", [])
    if 0 <= step_idx < len(steps):
        return steps[step_idx]["proactive_prompt"]
    return "I noticed you've been reflecting on this step. Would you like to review the next move together?"


def get_initial_whiteboard(topic_id: str, step_idx: int = 0) -> dict[str, Any]:
    """Return the initial whiteboard dictionary for a topic and step."""
    topic = get_topic(topic_id)
    steps = topic.get("steps", [])
    clamped_step = max(0, min(step_idx, len(steps) - 1)) if steps else 0
    step_data = steps[clamped_step] if steps else {}

    step_list = []
    for idx, s in enumerate(step_data.get("steps", [])):
        status = "pending"
        if idx < clamped_step:
            status = "completed"
        elif idx == clamped_step:
            status = "current"
        step_list.append(
            {
                "label": s["label"],
                "expr": s["expr"],
                "status": status,
            }
        )

    return {
        "title": step_data.get("whiteboard_title", topic["label"]),
        "problem": step_data.get("problem", ""),
        "steps": step_list,
        "socratic_hint": step_data.get("socratic_hint", ""),
        "visual_diagram": step_data.get("visual_diagram", "balance_scale"),
        "binary_choices": list(step_data.get("binary_choices", [])),
    }


def generate_offline_socratic_response(
    topic_id: str,
    step_idx: int,
    student_text: str,
    pedagogical_state: str,
    proactive: bool = False,
    attempt_num: int = 0,
) -> dict[str, Any]:
    """Generate deterministic Socratic response matching the LLM output schema."""
    topic = get_topic(topic_id)
    steps = topic.get("steps", [])
    clamped_step = max(0, min(step_idx, len(steps) - 1)) if steps else 0
    current_step = steps[clamped_step] if steps else {}

    whiteboard = get_initial_whiteboard(topic_id, clamped_step)
    text_lower = student_text.lower().strip()
    expected_kw = current_step.get("expected_keywords", [])

    is_correct = any(kw.lower() in text_lower for kw in expected_kw) if text_lower else False
    advance_step = False
    pedagogical_action = "socratic_hint"
    rot = attempt_num % 3

    if proactive:
        spoken_response = current_step.get(
            "proactive_prompt",
            "I noticed you're thinking carefully about this step. Want to talk through it together?",
        )
        pedagogical_action = "proactive_scaffold"

    elif pedagogical_state == "DISTRACTED_REENGAGE":
        responses = [
            f"Quick check-in! Let's bring our focus back to {current_step.get('problem', 'our problem')}. What do you think our next move is?",
            f"Hey there! Still with me? Take a quick look at the whiteboard and tell me what jumps out.",
            f"Let's pause and reset together. Ready to knock out this next step?",
        ]
        spoken_response = responses[rot]
        pedagogical_action = "reengage_focus"

    elif pedagogical_state == "FRUSTRATED_SCAFFOLD":
        choices = current_step.get("binary_choices", [])
        choice_text = (
            f" Would you rather {choices[0].lower()} or {choices[1].lower()}?"
            if len(choices) >= 2
            else ""
        )
        hint = current_step.get("socratic_hint", "")
        responses = [
            f"Totally fair, this step catches almost everyone! {hint}{choice_text}",
            f"Take a breath! You're doing great just working through it. {hint}{choice_text}",
            f"No worries at all, math can feel sticky here. Let's simplify: {choice_text or hint}",
        ]
        spoken_response = responses[rot]
        pedagogical_action = "scaffold_binary_choice"

    elif pedagogical_state == "CONFUSED_HINT":
        hint = current_step.get("socratic_hint", "")
        prefix = ""
        if "help" in text_lower or "hint" in text_lower or "what" in text_lower:
            prefix = "I've got your back! "
        responses = [
            f"{prefix}Here's a nudge to spot the pattern: {hint}",
            f"{prefix}Look closely at the numbers on this step. {hint}",
            f"{prefix}You're close! Think about what operation undoes what we have: {hint}",
        ]
        spoken_response = responses[rot]
        pedagogical_action = "socratic_hint"

    else:  # CONFIDENT_ADVANCE
        if is_correct:
            if clamped_step < len(steps) - 1:
                advance_step = True
                next_step = steps[clamped_step + 1]
                responses = [
                    f"Spot on! Great reasoning on that move. That brings us to: {next_step.get('problem', '')}. What's our plan here?",
                    f"Awesome insight! You nailed that. Now we're looking at: {next_step.get('problem', '')}. Where should we start?",
                    f"Yes, exactly! That simplifies everything nicely. On to: {next_step.get('problem', '')}. How should we approach it?",
                ]
                spoken_response = responses[rot]
                pedagogical_action = "advance_step"
                whiteboard = get_initial_whiteboard(topic_id, clamped_step + 1)
            else:
                spoken_response = (
                    "Outstanding work! You solved every single step with solid logic. "
                    "You've completely mastered this concept!"
                )
                pedagogical_action = "complete_problem"
                for s in whiteboard.get("steps", []):
                    s["status"] = "completed"
        else:
            hint = current_step.get("socratic_hint", "What does our problem tell us to do next?")
            responses = [
                f"Ooh, super close! Let's double-check: {hint}",
                f"You're definitely on the right track! Think about this: {hint}",
                f"Great attempt! Let's test that thought: {hint}",
            ]
            spoken_response = responses[rot]
            pedagogical_action = "elaborate_reasoning"

    return {
        "spoken_response": spoken_response,
        "whiteboard": whiteboard,
        "pedagogical_action": pedagogical_action,
        "advance_step": advance_step,
    }
