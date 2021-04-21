### 유도 과정

TPR = TP/(TP + FN)  
FPR = FP/(FP + TN)  

TP = TPR / (1 - TPR) * FN  
FP = FPR / (1 - FPR) * TN  


Jaccard = TP / (FN + TP + FP)  
FN + TP + FP = TP / Jaccard  
* TP = TPR / (1 - TPR) * FN이므로
FN + (TPR / (1 - TPR) * FN) + FP = ((TPR / (1 - TPR) * FN) / Jaccard)  
(1 + (TPR / (1 - TPR) - ((TPR / (1 - TPR) / Jaccard)))) * FN = - FP  
* alpha = (1 + (TPR / (1 - TPR) - ((TPR / (1 - TPR) / Jaccard))))라고 하면
FP = - alpha * FN  

area_diff = abs(FN - FP) / (FN + TP + FP + TN)  
area_diff = ((1 + alpha) * FN)  / (FN + (TPR / (1 - TPR) * FN) + (- alpha * FN) +  (- alpha * FN) * (1 - FPR) / FPR)  
area_diff = (1 + alpha) / (1 + (TPR / (1 - TPR)) + (- alpha) + (- alpha) * ((1 - FPR) / FPR))


```python

def cal_area_diff(Jaccard, TPR, FPR):
    alpha = (1 + (TPR / (1 - TPR) - ((TPR / (1 - TPR) / Jaccard))))
    area_diff = (1 + alpha) /\
                (1 + (TPR / (1 - TPR)) + (- alpha) * (1 + (1 - FPR) / FPR))
    return abs(area_diff)
    
    
muscle         = [0.8751, 0.9460, 0.0069]
subcutaneoust  = [0.8931, 0.9384, 0.0042]
viscera        = [0.7258, 0.8217, 0.0084]

abdomens  = [muscle, subcutaneoust, viscera]

for abdomen in abdomens:
    print(cal_area_diff(abdomen[0], abdomen[1], abdomen[2]))
   
    
#0.002120504004578214
#0.0008318447200486303
#0.0027597473330514917
```
