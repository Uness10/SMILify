
## 1. Data representation: add the animal/specimen dimension

Current:

```text
sample = V images of ONE animal
```

Prototype:

```text
sample = V images containing N known specimens
```

For single-view:

```text
(B, 3, H, W)
```

For multi-view:

```text
(B, V, 3, H, W)
```

with per-specimen annotations:

```text
specimen_i:
    2D keypoints
    pose
    shape
    translation
```

Crucially, each specimen has a **fixed identity/order**.

---

## 2. Shared ViT remains unchanged

Use **one ViT feature extractor** for all images.

Single view:

```text
Image → ViT → features
```

Multi-view:

```text
Image₁ ─┐
Image₂ ─┤
Image₃ ─┼→ shared ViT → stacked/fused features
...    ─┘
```

You don't need one ViT per animal or per camera.

---

## 3. Add `N−1` SMIL parameter heads

The existing architecture already has **one SMIL parameter head**.

For `N=3` animals:

```text
Existing head → specimen 1
New head      → specimen 2
New head      → specimen 3
```

All heads receive the shared image features.

Initially, the new heads can be **initialized by copying the pretrained single-animal head weights**.

---

## 4. Each head predicts one specimen

Instead of:

```text
features → θ
```

you now have:

```text
features → θ₁
         → θ₂
         → θ₃
```

where:

```text
θᵢ = poseᵢ + shapeᵢ + translationᵢ
```

Each parameter set goes through its own SMAL forward pass.

---

## 5. Strict specimen ↔ head correspondence

This is one of the biggest simplifications.

Define:

```text
Head 1 ↔ Specimen 1
Head 2 ↔ Specimen 2
Head 3 ↔ Specimen 3
```

throughout the footage/dataset.

Therefore:

```text
Prediction Head 1 ↔ GT Specimen 1
Prediction Head 2 ↔ GT Specimen 2
Prediction Head 3 ↔ GT Specimen 3
```

### Consequently, you do NOT need initially:

* Hungarian matching
* permutation-invariant loss
* DETR-style object queries
* identity discovery

This removes a major source of complexity.

---

## 6. Camera prediction becomes scene/view-level

Camera parameters should **not be duplicated per animal**.

Instead:

```text
Image features
      ↓
Camera head
      ↓
Camera parameters
```

while:

```text
Animal head 1 → pose₁, shape₁, trans₁
Animal head 2 → pose₂, shape₂, trans₂
Animal head 3 → pose₃, shape₃, trans₃
```

So:

```text
Scene/View-level:
    camera

Animal-level:
    pose
    shape
    translation
```

For multi-view, there will naturally be **one camera parameter set per camera/view**, not per animal.

---

## 7. SMAL runs once per specimen

The SMAL model itself remains shared/fixed.

```text
θ₁ → SMAL → Mesh₁
θ₂ → SMAL → Mesh₂
θ₃ → SMAL → Mesh₃
```

This can be implemented as a batched operation rather than maintaining three fundamentally different SMAL models.

---

## 8. Rendering/projection becomes multi-animal

Instead of one mesh:

```text
SMAL → Mesh → Camera → 2D
```

you have:

```text
Mesh₁ ─┐
Mesh₂ ─┼→ shared scene camera(s) → 2D projections
Mesh₃ ─┘
```

For multi-view:

```text
                 Mesh₁
                ↙  ↓  ↘
             Cam₁ Cam₂ Cam₃
```

and the **same three 3D animals** are projected into every camera.

You don't reconstruct a separate animal for each view.

---

## 9. Losses become per-specimen

Because the correspondence is fixed:

```text
L =
L(specimen₁_pred, specimen₁_GT)
+
L(specimen₂_pred, specimen₂_GT)
+
L(specimen₃_pred, specimen₃_GT)
```

with the existing SMILify losses applied independently to each specimen.

You can then aggregate/average across animals.

No permutation matching is needed.

---

## 10. Occlusion/visibility must become specimen-aware

The image can contain:

```text
Animal 1: fully visible
Animal 2: partially occluded
Animal 3: heavily occluded
```

The keypoint loss therefore needs to respect the existing visibility information:

```text
L₂D(animal_i)
    only uses valid/visible keypoints
```

This becomes particularly important once animals overlap.

For the first prototype, I wouldn't introduce sophisticated occlusion reasoning yet.

---

## 11. Multi-view should be a second step

I would structure development as:

### Prototype 1 — single view

```text
Image
 ↓
ViT
 ↓
N SMIL heads
 ↓
N × SMAL
 ↓
camera projection
 ↓
per-specimen losses
```

This isolates the fundamental multi-animal problem.

### Prototype 2 — multi-view

Then extend:

```text
V images
 ↓
shared ViT
 ↓
stack/fuse view features
 ↓
N SMIL heads
 ↓
N animals
 ↓
each animal projected into V cameras
```

The existing architecture already processes multiple views through the shared backbone and restores the view dimension. 

---

# Final architecture changes

| Component            | Current SMILify           | Multi-animal prototype                   |
| -------------------- | ------------------------- | ---------------------------------------- |
| Input                | 1 animal                  | N known specimens                        |
| ViT                  | 1 shared extractor        | **unchanged/shared**                     |
| Features             | One animal representation | Shared features for all specimens        |
| SMIL head            | 1                         | **N heads**                              |
| Initialization       | pretrained                | New heads copied from pretrained head    |
| Parameters           | one θ                     | θ₁ … θₙ                                  |
| Camera               | animal prediction         | **scene/view-level prediction**          |
| SMAL                 | one animal                | **N batched forward passes**             |
| Rendering            | one mesh                  | multiple meshes in same scene            |
| Matching             | implicit                  | **fixed head ↔ specimen correspondence** |
| Permutation matching | N/A                       | **not needed**                           |
| Occlusion            | single animal             | visibility per specimen/keypoint         |
| Multi-view           | existing support          | extend after single-view prototype       |

### In one sentence

**The core modification is: keep the shared ViT and SMILify machinery, add `N−1` copies of the existing parameter head, assign each head permanently to one known specimen, predict one set of animal parameters per head, run shared SMAL independently for each, and use the same scene camera(s) to project all reconstructed animals.**
