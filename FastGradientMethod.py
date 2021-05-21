import paddle


class FGM():
    """针对embedding层梯度上升干扰的对抗训练方法,Fast Gradient Method（FGM）"""

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:  # 检验参数是否可训练及范围
                self.backup[name] = param.numpy()  # 备份原有参数值
                grad_tensor = paddle.to_tensor(param.grad)  # param.grad是个numpy对象
                norm = paddle.norm(grad_tensor)  # norm化
                if norm != 0:
                    r_at = epsilon * grad_tensor / norm
                    param.add(r_at)  # 在原有embed值上添加向上梯度干扰

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                assert name in self.backup
                param.set_value(self.backup[name])  # 将原有embed参数还原
        self.backup = {}


#  使用样例：
# fgm = FGM(model)
# for batch_input, batch_label in data:
#     # 正常训练
#     loss = model(batch_input, batch_label)
#     loss.backward() # 反向传播，得到正常的grad
#     # 对抗训练
#     fgm.attack() # 在embedding上添加对抗扰动
#     loss_adv = model(batch_input, batch_label)
#     loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
#     fgm.restore() # 恢复embedding参数
#     # 梯度下降，更新参数
#     optimizer.step()
#     lr_scheduler.step()
#     optimizer.clear_gradients()