import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return float(self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p)


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        if self.model is None:
            raise ValueError("Model is not linked to optimizer")

        loss_history: list[float] = []
        self.iteration = 0

        loss_history.append(self.model.compute_loss())

        for _ in range(self.max_iter):
            delta = self._step()
            loss_history.append(self.model.compute_loss())

            if np.isnan(delta).any():
                break

            if float(np.sum(delta ** 2)) < self.tolerance:
                break

        self.model.loss_history = loss_history


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        w_old = self.model.w.copy()

        lr = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients()

        self.model.w = self.model.w - lr * gradient
        return self.model.w - w_old


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        if batch_size is None:
            raise ValueError("batch_size must be a positive integer")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = int(batch_size)

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects = X_train.shape[0]

        batch_indices = np.random.randint(0, num_objects, size=self.batch_size)
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        w_old = self.model.w.copy()
        lr = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(X_batch, y_batch)

        self.model.w = self.model.w - lr * gradient
        return self.model.w - w_old


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = int(batch_size)

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features), dtype=float)
            self.grad_sum = np.zeros(num_features, dtype=float)

        batch_indices = np.random.randint(0, num_objects, size=self.batch_size)

        for idx in batch_indices:
            new_grad = self.model.compute_gradients(X_train[idx:idx + 1], y_train[idx:idx + 1])
            old_grad = self.grad_memory[idx].copy()
            self.grad_sum += (new_grad - old_grad)
            self.grad_memory[idx] = new_grad

        avg_grad = self.grad_sum / num_objects

        w_old = self.model.w.copy()
        lr = self.lr_schedule.get_lr(self.iteration)
        self.model.w = self.model.w - lr * avg_grad
        return self.model.w - w_old


class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)

        w_old = self.model.w.copy()
        lr = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients()

        self.velocity = self.beta * self.velocity + lr * gradient
        self.model.w = self.model.w - self.velocity
        return self.model.w - w_old


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(self.model.w)
        if self.v is None:
            self.v = np.zeros_like(self.model.w)

        w_old = self.model.w.copy()
        lr = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients()

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradient ** 2)

        t = self.iteration + 1
        m_hat = self.m / (1.0 - (self.beta1 ** t))
        v_hat = self.v / (1.0 - (self.beta2 ** t))

        self.model.w = self.model.w - lr * (m_hat / (np.sqrt(v_hat) + self.eps))
        return self.model.w - w_old


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """
    def __init__(self):
        self.model = None
    

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        # не должна содержать непосредственных формул аналитического решения, за него ответственен другой объект
        if self.model is None:
            raise ValueError("Model is not linked to optimizer")

        X = getattr(self.model, "X_train", None)
        y = getattr(self.model, "y_train", None)
        if X is None or y is None:
            raise ValueError("Model has no training data; call model.fit(X, y) first")

        if getattr(self.model, "w", None) is None:
            self.model.w = np.zeros(X.shape[1], dtype=float)

        loss_history: list[float] = [self.model.compute_loss()]

        loss_fn = getattr(self.model, "loss_function", None)
        if loss_fn is None or not hasattr(loss_fn, "analytic_solution"):
            raise TypeError("Model loss_function does not support analytic_solution")

        self.model.w = loss_fn.analytic_solution(X, y)
        loss_history.append(self.model.compute_loss())
        self.model.loss_history = loss_history
