"""
Improved synthetic medical data generator for CHG validation.

Creates realistic multi-step medical reasoning chains with:
- Balanced correct/incorrect nodes
- Variable chain lengths
- Realistic error patterns
"""

import json
import random
import uuid
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.unified_schema import (
    ReasoningChain,
    ReasoningStep,
    DependencyGraph,
    Domain,
    ErrorType,
)


# Medical reasoning templates
MEDICAL_SCENARIOS = [
    {
        "query": "66-year-old male presents with chest pain, shortness of breath. What is the diagnosis?",
        "ground_truth": "Acute myocardial infarction (heart attack)",
        "correct_steps": [
            "Patient presents with substernal chest pain radiating to left arm",
            "ECG shows ST-segment elevation in leads II, III, aVF",
            "Troponin levels are significantly elevated (>0.4 ng/mL)",
            "Diagnosis: Acute inferior wall myocardial infarction",
            "Treatment: Emergency PCI (percutaneous coronary intervention)",
        ],
        "error_variations": [
            ("ECG shows normal sinus rhythm with no ST changes", 1, "factual"),
            ("Diagnosis: Gastroesophageal reflux disease (GERD)", 3, "logical"),
            ("Troponin levels are within normal range", 2, "factual"),
        ],
    },
    {
        "query": "28-year-old female with fever, productive cough, and dyspnea for 3 days. Diagnosis?",
        "ground_truth": "Community-acquired pneumonia",
        "correct_steps": [
            "Patient has fever (38.9°C), tachypnea, and crackles on auscultation",
            "Chest X-ray shows right lower lobe consolidation",
            "Sputum culture grows Streptococcus pneumoniae",
            "Diagnosis: Community-acquired bacterial pneumonia",
            "Treatment: Empiric antibiotics (amoxicillin-clavulanate)",
        ],
        "error_variations": [
            ("Chest X-ray is clear with no infiltrates", 1, "factual"),
            ("Diagnosis: Viral upper respiratory infection", 3, "logical"),
            ("Treatment: No antibiotics needed, supportive care only", 4, "logical"),
        ],
    },
    {
        "query": "45-year-old diabetic patient with polyuria, polydipsia, confusion. What's happening?",
        "ground_truth": "Diabetic ketoacidosis (DKA)",
        "correct": [
            "Blood glucose is 450 mg/dL",
            "Arterial blood gas shows pH 7.1 (acidosis)",
            "Urine ketones are strongly positive",
            "Diagnosis: Diabetic ketoacidosis",
            "Treatment: IV insulin and fluid resuscitation",
        ],
        "error_variations": [
            ("Blood glucose is 120 mg/dL (normal range)", 0, "factual"),
            ("pH is 7.4 (normal), no acidosis present", 1, "factual"),
            ("Diagnosis: Hyperosmolar hyperglycemic state (HHS)", 3, "logical"),
        ],
    },
    {
        "query": "72-year-old with sudden onset severe headache, neck stiffness. Diagnosis?",
        "ground_truth": "Subarachnoid hemorrhage",
        "correct_steps": [
            "Patient describes 'worst headache of my life', thunderclap onset",
            "CT scan shows blood in subarachnoid space",
            "Lumbar puncture confirms xanthochromia",
            "Diagnosis: Subarachnoid hemorrhage",
            "Immediate neurosurgical consultation required",
        ],
        "error_variations": [
            ("CT scan is normal with no intracranial bleeding", 1, "factual"),
            ("Diagnosis: Tension-type headache", 3, "logical"),
            ("Treatment: Over-the-counter analgesics and discharge home", 4, "logical"),
        ],
    },
]


def generate_reasoning_chain(
    scenario: dict, 
    introduce_error: bool = False,
    error_position: int = None,
    chain_length: int = None
) -> ReasoningChain:
    """Generate a single medical reasoning chain."""
    
    # Use scenario templates
    query = scenario["query"]
    ground_truth = scenario["ground_truth"]
    
    # Determine chain length
    if chain_length is None:
        chain_length = random.randint(3, 6)
    
    # Get correct steps
    if "correct_steps" in scenario:
        base_steps = scenario["correct_steps"][:chain_length]
    else:
        base_steps = scenario["correct"][:chain_length]
    
    # Build reasoning steps
    reasoning_steps = []
    error_introduced = False
    error_step_id = None
    
    if introduce_error and error_position is None:
        # Random error position (not first step)
        error_position = random.randint(1, len(base_steps) - 1)
    
    for i, step_text in enumerate(base_steps):
        is_correct = True
        is_origin = False
        error_type = None
        
        # Introduce error if needed
        if introduce_error and i == error_position and not error_introduced:
            # Pick a random error variation
            error_var = random.choice(scenario["error_variations"])
            error_text, expected_pos, err_type = error_var
            
            # Only use if position matches
            if expected_pos == i:
                step_text = error_text
                is_correct = False
                is_origin = True
                error_introduced = True
                error_step_id = i
                error_type = ErrorType.FACTUAL if err_type == "factual" else ErrorType.LOGICAL
        
        # Steps after error are contaminated
        if error_introduced and i > error_step_id:
            is_correct = False
            is_origin = False
            error_type = ErrorType.LOGICAL  # Propagated error
        
        reasoning_steps.append(
            ReasoningStep(
                step_id=i,
                text=step_text,
                is_correct=is_correct,
                is_origin=is_origin,
                error_type=error_type,
                depends_on=[i - 1] if i > 0 else [],
            )
        )
    
    # Build dependency graph (linear chain)
    nodes = list(range(len(reasoning_steps)))
    edges = [[i, i + 1] for i in range(len(reasoning_steps) - 1)]
    
    return ReasoningChain(
        domain=Domain.MEDICAL,
        query_id=str(uuid.uuid4()),
        query=query,
        ground_truth=ground_truth,
        reasoning_steps=reasoning_steps,
        dependency_graph=DependencyGraph(nodes=nodes, edges=edges),
    )


def generate_synthetic_dataset(
    num_samples: int = 1000,
    error_rate: float = 0.4,
    output_path: Path = None,
) -> List[ReasoningChain]:
    """
    Generate synthetic medical dataset.
    
    Args:
        num_samples: Number of chains to generate
        error_rate: Proportion of chains with errors (default 40%)
        output_path: Where to save the dataset
    
    Returns:
        List of ReasoningChains
    """
    
    chains = []
    num_errors = int(num_samples * error_rate)
    num_correct = num_samples - num_errors
    
    print(f"Generating {num_samples} synthetic medical chains...")
    print(f"  - Correct chains: {num_correct}")
    print(f"  - Error chains: {num_errors}")
    
    # Generate correct chains
    for i in range(num_correct):
        scenario = random.choice(MEDICAL_SCENARIOS)
        chain = generate_reasoning_chain(scenario, introduce_error=False)
        chains.append(chain)
    
    # Generate error chains
    for i in range(num_errors):
        scenario = random.choice(MEDICAL_SCENARIOS)
        chain = generate_reasoning_chain(scenario, introduce_error=True)
        chains.append(chain)
    
    # Shuffle
    random.shuffle(chains)
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for chain in chains:
                f.write(chain.to_json(pretty=False) + "\n")
        
        # Print statistics
        total_nodes = sum(len(c.reasoning_steps) for c in chains)
        correct_nodes = sum(
            sum(1 for s in c.reasoning_steps if s.is_correct) for c in chains
        )
        incorrect_nodes = total_nodes - correct_nodes
        origin_nodes = sum(
            sum(1 for s in c.reasoning_steps if s.is_origin) for c in chains
        )
        
        print(f"\n=== Dataset Statistics ===")
        print(f"Total chains: {len(chains)}")
        print(f"Total nodes: {total_nodes}")
        print(f"Correct nodes: {correct_nodes} ({correct_nodes/total_nodes*100:.1f}%)")
        print(f"Incorrect nodes: {incorrect_nodes} ({incorrect_nodes/total_nodes*100:.1f}%)")
        print(f"Origin nodes: {origin_nodes}")
        print(f"Saved to: {output_path}")
    
    return chains


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "data/processed/medical_synthetic_improved.jsonl"
    
    # Generate 1000 samples with 40% error rate
    chains = generate_synthetic_dataset(
        num_samples=1000,
        error_rate=0.4,
        output_path=output_path,
    )
    
    print("\n✅ Synthetic medical data generation complete!")
