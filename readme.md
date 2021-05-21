## Adversarial Training for NLP in Paddle

An attempt at replicating the Fast Gradient Method（FGM) of  adversarial training for NLP in paddle.

### Usage Example


```code
from FastGradientMethod import FGM
...

fgm = FGM(model)
for batch_input, batch_label in data:
     loss = model(batch_input, batch_label)
     loss.backward()
     fgm.attack()
     loss_adv = model(batch_input, batch_label)
     loss_adv.backward()
     fgm.restore() 
     optimizer.step()
     lr_scheduler.step()
     optimizer.clear_gradients()
```


### Reference
https://fyubang.com/2019/10/15/adversarial-train/