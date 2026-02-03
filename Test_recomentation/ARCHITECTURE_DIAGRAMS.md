# Test Recommendation System - Visual Guide

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT DATA                                 │
├─────────────────────────────────────────────────────────────────────┤
│  • patient-test.csv: Patient test records (160K rows)              │
│  • test-disease-organ.csv: Test metadata (183 tests)               │
│  • patient-one-hot-labeled-disease.csv: Disease labels (24K pts)   │
└────────────────────────────────┬────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   HETEROGENEOUS GRAPH                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    ┌──────────┐                                                    │
│    │ PATIENT  │ ──────had_test──────→ ┌──────────┐               │
│    │  (24K)   │                        │ LAB_TEST │               │
│    └──────────┘                        │  (183)   │               │
│                                        └─────┬────┘               │
│                                              │                     │
│                        ┌────tests_organ──────┤                    │
│                        │                     │                     │
│                        ↓                     │                     │
│                   ┌────────┐                 │                     │
│                   │ ORGAN  │                 │                     │
│                   │  (12)  │←──occurs_in──┐  │                     │
│                   └────────┘               │  │                     │
│                                            │  │                     │
│                                            │  └──associated_with──→ │
│                                       ┌────┴────┐                  │
│                                       │ DISEASE │                  │
│                                       │  (44)   │                  │
│                                       └─────────┘                  │
│                                                                     │
│  Node Types: 4 (patient, lab_test, organ, disease)                │
│  Edge Types: 4 (had_test, tests_organ, associated_with, occurs_in)│
└─────────────────────────────────┬───────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        GNN MODEL (pyHGT)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: Node features (5-dim) + Graph structure                    │
│         ↓                                                           │
│  ┌──────────────────┐                                              │
│  │ Heterogeneous    │  - Transforms features to hidden space       │
│  │ Adaptation       │  - Separate transform per node type          │
│  └────────┬─────────┘                                              │
│           ↓                                                         │
│  ┌──────────────────┐                                              │
│  │ HGT Conv Layer 1 │  - Message passing with attention            │
│  │ (64-dim, 4 heads)│  - Relation-aware aggregation                │
│  └────────┬─────────┘                                              │
│           ↓                                                         │
│  ┌──────────────────┐                                              │
│  │ HGT Conv Layer 2 │  - Deeper graph context                      │
│  │ (64-dim, 4 heads)│  - Refined embeddings                        │
│  └────────┬─────────┘                                              │
│           ↓                                                         │
│  Output: Node embeddings (64-dim)                                  │
│          - Patient embeddings: [24K, 64]                           │
│          - Test embeddings: [183, 64]                              │
│          - Organ embeddings: [12, 64]                              │
│          - Disease embeddings: [44, 64]                            │
│                                                                     │
└──────────────────┬──────────────────────────────┬───────────────────┘
                   ↓                              ↓
┌─────────────────────────────┐   ┌──────────────────────────────────┐
│      CLASSIFIER             │   │    MATCHER (Link Predictor)      │
│   (Disease Prediction)      │   │    (Test Recommendation)         │
├─────────────────────────────┤   ├──────────────────────────────────┤
│                             │   │                                  │
│  Input: Patient embedding   │   │  Input: Patient emb + Test emb   │
│         [1, 64]             │   │         [1, 64]   +   [1, 64]    │
│         ↓                   │   │         ↓              ↓          │
│  ┌───────────────┐          │   │  ┌──────────┐   ┌──────────┐    │
│  │ Linear Layer  │          │   │  │ Left     │   │ Right    │    │
│  │ (64 → 44)     │          │   │  │ Linear   │   │ Linear   │    │
│  └───────┬───────┘          │   │  └────┬─────┘   └────┬─────┘    │
│          ↓                  │   │       ↓              ↓          │
│  ┌───────────────┐          │   │  ┌──────────────────────┐       │
│  │   Sigmoid     │          │   │  │ Attention Score      │       │
│  └───────┬───────┘          │   │  │ score = (left·right) │       │
│          ↓                  │   │  │       / sqrt(64)     │       │
│  Disease probabilities      │   │  └──────────┬───────────┘       │
│  [44 diseases]              │   │             ↓                   │
│  e.g., [0.82, 0.45, ...]    │   │  Link score (0 to 1)            │
│                             │   │  0.89 = High compatibility      │
│                             │   │  0.12 = Low compatibility       │
│                             │   │                                  │
└──────────────┬──────────────┘   └────────────┬─────────────────────┘
               ↓                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                   TEST RECOMMENDER MODULE                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 1: Calculate Confidence                                       │
│  ┌────────────────────────────────────────────┐                    │
│  │ max_confidence = max(disease_probs)        │                    │
│  │ mean_confidence = mean(disease_probs)      │                    │
│  │ entropy = -Σ p·log(p)                      │                    │
│  └────────────────────────────────────────────┘                    │
│           ↓                                                         │
│  Step 2: Check Threshold                                           │
│  ┌────────────────────────────────────────────┐                    │
│  │ if max_confidence < 0.7:                   │                    │
│  │     proceed with recommendations           │                    │
│  │ else:                                      │                    │
│  │     return "High confidence, no tests"     │                    │
│  └────────────────────────────────────────────┘                    │
│           ↓                                                         │
│  Step 3: Compute Link Scores                                       │
│  ┌────────────────────────────────────────────┐                    │
│  │ For each test in all_tests:                │                    │
│  │     score[test] = Matcher(patient, test)   │                    │
│  │ Result: [0.89, 0.87, 0.12, ...]            │                    │
│  └────────────────────────────────────────────┘                    │
│           ↓                                                         │
│  Step 4: Filter Already-Taken Tests                                │
│  ┌────────────────────────────────────────────┐                    │
│  │ taken_tests = get_patient_tests(patient)   │                    │
│  │ available = all_tests - taken_tests        │                    │
│  │ scores = scores[available]                 │                    │
│  └────────────────────────────────────────────┘                    │
│           ↓                                                         │
│  Step 5: Rank and Return Top-K                                     │
│  ┌────────────────────────────────────────────┐                    │
│  │ recommendations = top_k(scores, k=5)       │                    │
│  │ Add context (organs, diseases)             │                    │
│  └────────────────────────────────────────────┘                    │
│                                                                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Patient ID: 139760                                                 │
│  ───────────────────────────────────────────                        │
│                                                                      │
│  Disease Predictions:                                               │
│    • chronic kidney disease: 0.456                                  │
│    • diabetes mellitus: 0.387                                       │
│    • anemia: 0.334                                                  │
│                                                                      │
│  Confidence: 0.456 (LOW) ← Triggers recommendations                │
│                                                                      │
│  Recommended Tests:                                                 │
│    1. Creatinine Result (score: 0.892)                             │
│       → Related: kidney (organ), CKD (disease)                      │
│    2. Blood Urea Result (score: 0.871)                             │
│       → Related: kidney (organ), kidney damage (disease)            │
│    3. GFR (score: 0.834)                                           │
│       → Related: kidney (organ), renal failure (disease)            │
│    4. Serum Sodium (score: 0.782)                                  │
│       → Related: kidney (organ), fluid imbalance (disease)          │
│    5. Serum Potassium (score: 0.756)                               │
│       → Related: kidney (organ), electrolyte disorder (disease)     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Training Process Flow

```
┌──────────────────┐
│  Load CSV Data   │
│  • patient-test  │
│  • test-disease  │
│  • labels        │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Build Graph     │
│  • Add nodes     │
│  • Add edges     │
│  • Extract feats │
└────────┬─────────┘
         ↓
┌──────────────────────────────────┐
│  Create Training Data            │
│  ┌────────────────────────────┐  │
│  │ Positive Links (Actual)    │  │
│  │ (Patient_A, Test_B) = 1    │  │
│  │ 160K pairs from data       │  │
│  └────────────────────────────┘  │
│  ┌────────────────────────────┐  │
│  │ Negative Links (Sampled)   │  │
│  │ (Patient_A, Test_C) = 0    │  │
│  │ 160K random pairs          │  │
│  └────────────────────────────┘  │
│  Split: 70% / 15% / 15%          │
└────────┬─────────────────────────┘
         ↓
┌───────────────────────────────────┐
│  Training Loop (20 epochs)        │
│                                   │
│  For each batch:                  │
│  ┌─────────────────────────────┐  │
│  │ 1. Sample pos/neg links     │  │
│  │ 2. Get embeddings via GNN   │  │
│  │ 3. Compute link scores      │  │
│  │ 4. Calculate BCE loss       │  │
│  │ 5. Backprop & update        │  │
│  └─────────────────────────────┘  │
│                                   │
│  After each epoch:                │
│  ┌─────────────────────────────┐  │
│  │ • Evaluate on validation    │  │
│  │ • Compute AUC, AP, Acc      │  │
│  │ • Print metrics             │  │
│  └─────────────────────────────┘  │
│                                   │
└────────┬──────────────────────────┘
         ↓
┌────────────────────┐
│  Final Evaluation  │
│  • Test AUC > 0.75 │
│  • Test AP > 0.70  │
│  • Test Acc > 0.70 │
└────────┬───────────┘
         ↓
┌────────────────────┐
│  Save Model        │
│  • GNN weights     │
│  • Matcher weights │
│  • Classifier      │
│  • History         │
└────────────────────┘
```

## Inference Process Flow

```
┌──────────────────┐
│  Load Model      │
│  • GNN           │
│  • Matcher       │
│  • Classifier    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Load Graph      │
│  (Same as train) │
└────────┬─────────┘
         ↓
┌───────────────────────────────┐
│  Input: Patient ID            │
│  e.g., "139760"               │
└────────┬──────────────────────┘
         ↓
┌───────────────────────────────┐
│  Get Patient Embedding        │
│  1. Sample subgraph           │
│  2. Extract features          │
│  3. Run GNN                   │
│  4. Get patient vector [64]   │
└────────┬──────────────────────┘
         ↓
┌───────────────────────────────┐
│  Predict Diseases             │
│  1. Pass through classifier   │
│  2. Apply sigmoid             │
│  3. Get probabilities [44]    │
└────────┬──────────────────────┘
         ↓
┌───────────────────────────────┐
│  Calculate Confidence         │
│  max_conf = max(probs)        │
│  Example: 0.456               │
└────────┬──────────────────────┘
         ↓
         ┌─ Is confidence < 0.7?
         │
    YES  │  NO
    ↓    │  ↓
┌────────────────┐  ┌───────────────────┐
│ Get All Test   │  │ Return:           │
│ Embeddings     │  │ "High confidence, │
│ [183, 64]      │  │  no tests needed" │
└───────┬────────┘  └───────────────────┘
        ↓
┌────────────────────┐
│ Compute Link       │
│ Scores [183]       │
│ Via Matcher        │
└───────┬────────────┘
        ↓
┌────────────────────┐
│ Filter Taken Tests │
│ Available: [102]   │
│ (patient took 81)  │
└───────┬────────────┘
        ↓
┌────────────────────┐
│ Rank by Score      │
│ Sort descending    │
└───────┬────────────┘
        ↓
┌────────────────────┐
│ Select Top-K       │
│ Default: K=5       │
└───────┬────────────┘
        ↓
┌────────────────────────────┐
│ Add Context Info           │
│ • Test → Organ mapping     │
│ • Test → Disease mapping   │
│ • Explanation text         │
└───────┬────────────────────┘
        ↓
┌────────────────────────────┐
│ Return Recommendations     │
│ [                          │
│   {                        │
│     "test": "Creatinine",  │
│     "score": 0.892,        │
│     "organs": ["kidney"],  │
│     "diseases": ["CKD"]    │
│   },                       │
│   ...                      │
│ ]                          │
└────────────────────────────┘
```

## Data Flow Example

```
Patient 139760 enters system
        ↓
Has taken 15 tests already:
  - Blood Glucose
  - HbA1c
  - Cholesterol
  - ...
        ↓
System builds subgraph:
  Patient_139760 ─── had_test ──→ Blood_Glucose ─── tests_organ ──→ Pancreas
                                                  └─ associated_with ─→ Diabetes
        ↓
GNN processes subgraph:
  Patient_139760 → [0.12, -0.45, 0.78, ..., 0.34] (64 numbers)
        ↓
Classifier predicts diseases:
  Diabetes: 0.823  ← High confidence
  CKD: 0.456       ← Medium confidence  
  Anemia: 0.334    ← Low confidence
        ↓
Max confidence = 0.823... Wait, that's high!
No need for recommendations in this case.

───────────────────────────────

Another Patient 200041:
        ↓
Has taken 3 tests only:
  - Blood Glucose (slightly high)
  - Cholesterol (normal)
  - WBC (slightly elevated)
        ↓
Classifier predicts:
  Diabetes: 0.387  
  Infection: 0.298
  Thyroid: 0.245
        ↓
Max confidence = 0.387 ← LOW! Trigger recommendations
        ↓
Compute link scores for all 183 tests:
  HbA1c: 0.892
  Fasting Glucose: 0.871
  OGTT: 0.834
  TSH: 0.782
  ...
        ↓
Filter out already taken (3 tests):
  Available: 180 tests
        ↓
Recommend top 5:
  1. HbA1c (diabetes confirmation)
  2. Fasting Glucose (diabetes)
  3. OGTT (diabetes)
  4. TSH (thyroid check)
  5. CRP (infection marker)
        ↓
Output to doctor:
  "Patient shows signs of possible diabetes (38.7% confidence).
   Recommend these tests to improve diagnostic certainty."
```

## File Organization

```
pyHGT-implementation/
│
├── data/                           # Data files (existing)
│   ├── patient-test.csv
│   ├── test-disease-organ.csv
│   └── patient-one-hot-labeled-disease.csv
│
├── pyHGT/                          # pyHGT library (existing)
│   ├── __init__.py
│   ├── conv.py
│   ├── data.py
│   ├── model.py
│   └── utils.py
│
├── train.py                        # Original disease prediction (UNCHANGED)
│
├── test_recommender.py             # NEW: Core recommendation logic
│   └── TestRecommender class
│       ├── recommend_tests()
│       ├── get_disease_confidence()
│       ├── filter_available_tests()
│       └── explain_recommendation()
│
├── train_test_recommendation.py    # NEW: Training script
│   ├── Load data & build graph
│   ├── Create training examples
│   ├── Train GNN + Matcher
│   └── Save model
│
├── recommend_tests.py              # NEW: Inference script
│   ├── Load trained model
│   ├── Make recommendations
│   └── Demo mode
│
├── models_saved/                   # Model checkpoints
│   ├── trained_model.pth          # Existing disease model
│   └── test_recommendation_model.pth  # NEW: Test rec model
│
├── TEST_RECOMMENDATION_GUIDE.md    # NEW: Detailed documentation
├── QUICK_START.md                  # NEW: Quick reference
├── IMPLEMENTATION_SUMMARY.md       # NEW: Implementation overview
└── ARCHITECTURE_DIAGRAMS.md        # NEW: This file
```

---

**Understanding This System:**

1. **Start with the architecture diagram** to see how components connect
2. **Follow the training process** to understand how the model learns
3. **Trace the inference flow** to see how recommendations are made
4. **Study the data flow example** for concrete understanding
5. **Reference file organization** to navigate the codebase

**Key Insight:** Link prediction learns patterns like "patients with diabetes often need HbA1c test" and "patients with kidney issues benefit from creatinine test", then applies these patterns to recommend tests for new patients with similar characteristics.
