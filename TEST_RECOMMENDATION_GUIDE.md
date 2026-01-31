# Test Recommendation System - Implementation Guide

## Overview

This implementation adds **test recommendation functionality** to the existing medical diagnosis system using **link prediction** in the pyHGT heterogeneous graph neural network.

### What Problem Does This Solve?

When a patient enters the healthcare system with some initial tests, the system:
1. Predicts potential diseases based on available test results
2. Calculates confidence levels for these predictions
3. **If confidence is low** (indicating uncertainty), recommends additional tests that would be most informative
4. Uses link prediction to identify which tests would provide the most valuable information for diagnosis

### Real-World Use Case

**Scenario:** A new patient visits with symptoms and takes basic blood tests. 

- The system predicts possible diseases but with low confidence (e.g., 0.45)
- Instead of making uncertain diagnoses, it recommends specific additional tests
- Example: If kidney disease is suspected but uncertain, recommend "Creatinine", "Blood Urea", "GFR" tests
- After these tests, confidence improves and diagnosis becomes more accurate

**Key Insight:** This is particularly useful for new patients or when initial symptoms are ambiguous.

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                   Heterogeneous Graph                       │
│                                                             │
│  ┌─────────┐      ┌──────────┐      ┌────────┐           │
│  │ Patient │─────→│ Lab Test │─────→│ Organ  │           │
│  └─────────┘      └──────────┘      └────────┘           │
│       │                │                   │               │
│       │                └──────→┌─────────┐│               │
│       └───────────────────────→│ Disease │←───────────    │
│                                └─────────┘                │
└─────────────────────────────────────────────────────────────┘
                           ↓
            ┌──────────────────────────────┐
            │      pyHGT GNN Model         │
            │  (Extracts embeddings for    │
            │   patients, tests, organs)   │
            └──────────────────────────────┘
                     ↓             ↓
        ┌────────────────┐   ┌──────────────────┐
        │   Classifier   │   │  Link Predictor  │
        │  (Disease      │   │  (Matcher Model) │
        │   Prediction)  │   │                  │
        └────────────────┘   └──────────────────┘
                ↓                      ↓
        ┌──────────────────────────────────────┐
        │    Test Recommender Module           │
        │  - Evaluates confidence              │
        │  - Filters available tests           │
        │  - Ranks recommendations             │
        └──────────────────────────────────────┘
```

### Key Models

1. **GNN (Graph Neural Network)**: 
   - Learns embeddings for all nodes (patients, tests, organs, diseases)
   - Captures relationships in the heterogeneous graph
   - Uses HGT (Heterogeneous Graph Transformer) convolution

2. **Matcher (Link Predictor)**:
   - Predicts likelihood of a link between patient and test
   - Uses attention mechanism to compute compatibility scores
   - Already existed in pyHGT, we adapt it for test recommendations

3. **Classifier**:
   - Multi-label disease prediction
   - Outputs probability for each disease
   - Used to assess confidence

4. **TestRecommender**:
   - New module created for this feature
   - Evaluates prediction confidence
   - Recommends tests based on link scores
   - Filters out already-taken tests

---

## Implementation Details

### Files Created (NO EXISTING CODE MODIFIED)

1. **`test_recommender.py`**: Core recommendation logic
   - `TestRecommender` class
   - Confidence calculation
   - Test filtering and ranking
   - Explanation generation

2. **`train_test_recommendation.py`**: Training script
   - Trains link prediction model
   - Uses positive/negative sampling
   - Evaluates with AUC, AP, Accuracy metrics
   - Saves trained model

3. **`recommend_tests.py`**: Inference script
   - Loads trained model
   - Recommends tests for patients
   - Demo mode for testing
   - Command-line interface

4. **`TEST_RECOMMENDATION_GUIDE.md`**: This documentation

### Training Process

The model is trained using **supervised link prediction**:

#### Positive Examples
- Actual patient-test pairs from the data
- If Patient A took Test B, this is a positive link

#### Negative Examples
- Randomly sampled patient-test pairs that don't exist
- Patient A did NOT take Test C → negative example

#### Training Objective
```
Maximize score for positive links
Minimize score for negative links
```

#### Loss Function
```python
Binary Cross-Entropy Loss
- Positive pairs should have score ≈ 1
- Negative pairs should have score ≈ 0
```

### Link Prediction Mechanics

The Matcher model computes a **compatibility score** between patient and test embeddings:

```python
# Patient embedding from GNN
patient_emb = GNN(patient_subgraph)  # Shape: [1, hidden_dim]

# Test embedding from GNN  
test_emb = GNN(test_subgraph)  # Shape: [1, hidden_dim]

# Link score (higher = more compatible)
score = Matcher(patient_emb, test_emb)  # Scalar value

# Convert to probability
prob = sigmoid(score)  # Value between 0 and 1
```

### Recommendation Algorithm

```python
def recommend(patient):
    # Step 1: Get disease predictions
    disease_probs = Classifier(GNN(patient))
    confidence = max(disease_probs)
    
    # Step 2: Check if recommendation needed
    if confidence < threshold (e.g., 0.7):
        
        # Step 3: Get all test embeddings
        all_test_embs = [GNN(test) for test in all_tests]
        
        # Step 4: Compute link scores
        scores = [Matcher(patient_emb, test_emb) 
                  for test_emb in all_test_embs]
        
        # Step 5: Filter already-taken tests
        available_tests = [test for test in all_tests 
                          if not taken_by(patient, test)]
        
        # Step 6: Rank by score
        recommended = top_k(available_tests, scores, k=5)
        
        return recommended
    else:
        return []  # High confidence, no tests needed
```

---

## Usage Guide

### 1. Training the Model

```bash
cd /Users/charlie/Documents/Coding/VS\ Code/Language_python/FYP/New_2026/pyHGT-implementation

python train_test_recommendation.py
```

**What happens:**
- Loads patient-test data
- Builds heterogeneous graph
- Trains GNN + Matcher for link prediction
- Evaluates on validation/test sets
- Saves model to `models_saved/test_recommendation_model.pth`
- Saves training history to `models_saved/test_recommendation_history.json`

**Expected training time:** 10-30 minutes depending on hardware

**Expected metrics:**
- Link Prediction AUC: > 0.75 (good)
- Link Prediction AP: > 0.70 (good)
- Accuracy: > 0.70 (good)

### 2. Making Recommendations

#### For a specific patient:
```bash
python recommend_tests.py --patient_id 139760 --top_k 5
```

#### Demo mode (random patients):
```bash
python recommend_tests.py --demo --top_k 5
```

**Output example:**
```
Patient ID: 139760
======================================================================

--- Disease Predictions ---
  chronic kidney disease: 0.823
  diabetes mellitus: 0.456
  anemia: 0.387

--- Test Recommendations ---
Confidence: 0.456
Needs recommendation: True

Recommended Tests:
  1. Creatinine Result (score: 0.892)
  2. Blood Urea Result (score: 0.871)
  3. GFR (score: 0.834)
  4. Serum Sodium (score: 0.782)
  5. Serum Potassium (score: 0.756)
```

### 3. Integration with Existing Disease Prediction

You can load the test recommendation model alongside your disease prediction model:

```python
from pyHGT.model import GNN, Matcher
from test_recommender import TestRecommender
import torch

# Load test recommendation model
checkpoint = torch.load('models_saved/test_recommendation_model.pth')
gnn.load_state_dict(checkpoint['gnn_state_dict'])
matcher.load_state_dict(checkpoint['matcher_state_dict'])

# Create recommender
recommender = TestRecommender(matcher, confidence_threshold=0.7)

# For a patient
patient_emb = get_patient_embedding(patient_id)
disease_probs = classifier(patient_emb)

# Get recommendations
recommendations = recommender.recommend_tests(
    patient_emb,
    all_test_embeddings,
    [patient_id],
    disease_probs,
    graph,
    top_k=5
)
```

---

## Technical Deep Dive

### Why Link Prediction Works Here

**Graph Structure Insight:**
- Patients who take certain test combinations often have similar diseases
- Tests connected to similar organs/diseases are diagnostically related
- The GNN learns these patterns from the graph structure

**Example:**
```
Patient A → [Glucose Test, HbA1c Test] → Diabetes
Patient B → [Glucose Test, ?] → Suspected Diabetes (low confidence)

Link Prediction: HbA1c Test has high compatibility with Patient B
Recommendation: Suggest HbA1c Test to Patient B
```

### Confidence Metrics

We use three complementary metrics:

1. **Max Confidence**: `max(disease_probabilities)`
   - Single highest disease probability
   - Simple but effective

2. **Mean Confidence**: `mean(disease_probabilities)`
   - Average across all diseases
   - Captures overall certainty

3. **Entropy**: `-Σ p·log(p) + (1-p)·log(1-p)`
   - Information theory measure
   - Higher entropy = more uncertainty
   - Lower entropy = more confident

### Filtering Already-Taken Tests

The system automatically excludes tests the patient has already taken:

```python
# Check patient's edges in graph
taken_tests = set()
for edge in patient.edges:
    if edge.type == "had_test":
        taken_tests.add(edge.target_test)

# Only recommend new tests
available_tests = all_tests - taken_tests
```

### Graph Node Types and Edges

**Node Types:**
- `patient`: Individual patients
- `lab_test`: Medical tests (blood tests, scans, etc.)
- `organ`: Body organs (kidney, liver, heart, etc.)
- `disease`: Medical conditions

**Edge Types:**
- `patient → lab_test` (had_test): Patient took this test
- `lab_test → organ` (tests_organ): Test examines this organ
- `lab_test → disease` (associated_with): Test relevant for this disease
- `disease → organ` (occurs_in): Disease affects this organ

### Feature Engineering

**Patient Features (5-dimensional):**
1. `num_tests`: Total tests taken
2. `mean_abn`: Average abnormality score
3. `max_abn`: Maximum abnormality
4. `min_abn`: Minimum abnormality
5. `last_time`: Most recent test date

**Test Features (5-dimensional, sparse):**
1. `low`: Lower threshold for normal range
2. `high`: Upper threshold for normal range
3-5. Padding zeros

**Organ/Disease Features:**
- Zero vectors (rely on graph structure)

---

## Evaluation Metrics

### Link Prediction Metrics

1. **AUC-ROC (Area Under Curve)**
   - Measures ability to distinguish positive from negative links
   - Range: 0.5 (random) to 1.0 (perfect)
   - Good: > 0.75

2. **Average Precision (AP)**
   - Precision-recall curve area
   - Emphasizes top-ranked recommendations
   - Good: > 0.70

3. **Accuracy**
   - Percentage of correct predictions
   - Threshold: 0.5
   - Good: > 0.70

### Recommendation Quality (Manual Evaluation)

- **Relevance**: Do recommended tests relate to suspected diseases?
- **Diversity**: Are recommendations from different organ systems?
- **Coverage**: Can the system recommend across all disease types?

---

## Advantages of This Approach

### 1. **Leverages Existing Architecture**
- Uses the Matcher model already in pyHGT
- No need to design new neural network
- Proven link prediction approach

### 2. **Interpretable Recommendations**
- Can trace why test was recommended
- Based on graph structure (organs, diseases)
- Confidence scores provide transparency

### 3. **Handles New Patients**
- Works even with minimal initial tests
- Graph structure provides prior knowledge
- Can recommend informative first tests

### 4. **Scalable**
- Efficient inference with cached embeddings
- Can rank thousands of tests quickly
- Batch processing for multiple patients

### 5. **Domain-Aware**
- Incorporates medical knowledge (test-organ-disease relations)
- Not just data-driven, also knowledge-driven
- Respects clinical relationships

---

## Limitations and Future Work

### Current Limitations

1. **Cold Start**: Brand new patients with zero tests are challenging
2. **Test Costs**: Doesn't consider economic factors of tests
3. **Test Availability**: Doesn't check if test is available at facility
4. **Temporal Dynamics**: Doesn't model disease progression over time
5. **Multi-Step Planning**: Recommends one batch, not sequential strategy

### Potential Improvements

1. **Cost-Benefit Analysis**
   ```python
   score = link_score - cost_weight * test_cost
   ```

2. **Sequential Recommendations**
   - Recommend test A
   - Patient takes test A
   - Update graph and re-recommend
   - Adaptive testing strategy

3. **Uncertainty Quantification**
   - Bayesian neural networks
   - Confidence intervals on recommendations
   - Risk-aware recommendations

4. **Reinforcement Learning**
   - Learn optimal test ordering policy
   - Maximize diagnostic gain per test
   - Minimize total tests needed

5. **Multi-Modal Features**
   - Include patient demographics (age, gender)
   - Symptom descriptions (NLP)
   - Medical imaging (CNN)

---

## Troubleshooting

### Issue: Model doesn't train (loss is NaN)

**Solution:**
- Check for missing values in data
- Reduce learning rate: `lr=0.0001`
- Add gradient clipping: `torch.nn.utils.clip_grad_norm_(params, 1.0)`

### Issue: All recommendations are the same tests

**Solution:**
- Model may have collapsed
- Try increasing negative sampling ratio
- Add regularization: `weight_decay=1e-4`
- Check if graph has sufficient diversity

### Issue: Confidence always low/high

**Solution:**
- Adjust `confidence_threshold` parameter
- Check classifier calibration
- May need temperature scaling on disease predictions

### Issue: Recommends already-taken tests

**Solution:**
- Bug in filtering logic
- Check `filter_available_tests()` function
- Verify graph edge structure is correct

---

## Key Takeaways

### What We Built
✅ Link prediction model for patient-test recommendations  
✅ Confidence-based recommendation trigger  
✅ Automatic filtering of taken tests  
✅ Integration with existing disease prediction  
✅ Complete training and inference pipeline  

### What Makes It Work
🔑 Heterogeneous graph captures medical knowledge  
🔑 GNN learns meaningful embeddings  
🔑 Link prediction identifies informative tests  
🔑 Confidence metrics guide when to recommend  

### How It Helps
💡 Assists doctors in test ordering decisions  
💡 Reduces diagnostic uncertainty  
💡 Particularly valuable for complex/ambiguous cases  
💡 Provides explainable recommendations  

---

## Contact and Support

For questions or issues with this implementation, please check:
1. Training logs in `models_saved/test_recommendation_history.json`
2. Model checkpoint in `models_saved/test_recommendation_model.pth`
3. Error messages from training/inference scripts

---

**Implementation Date:** January 2026  
**Branch:** `feature-test-recom`  
**Status:** ✅ Complete and Ready for Training
