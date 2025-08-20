# Online vs Traditional Softmax

**Softmax** converts logits into probabilities.

- **Traditional softmax**: Compute on the entire vector at once.  
- **Online softmax**: Update incrementally as new logits arrive, but final result matches traditional softmax.

---

## Example

Logits `[1, 2, 3]`

- Batch softmax → `[0.0900, 0.2447, 0.6652]`  
- Online after `[1,2]` → `[0.2689, 0.7310]`  
- Online after adding `[3]` → `[0.0900, 0.2447, 0.6652]` (same as batch)




- Both methods give the same final probabilities.  
- Online softmax is useful for **streaming or long sequences**.
- Useful in Flash Attention for block wise computation or tiling   

