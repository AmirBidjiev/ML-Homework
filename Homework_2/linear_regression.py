import numpy as np 
from interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface, AbstractOptimizer
from descents import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC



class MSELoss(LossFunction, LossFunctionClosedFormMixin):

    def __init__(self, analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

        

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: float, значение MSE на данных X,y для весов w
        """
        y_pred = X @ w
        err = y_pred - y
        return float(np.sum(err ** 2) / len(y))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: np.ndarray, численный градиент MSE в точке w
        """
        y_pred = X @ w
        err = y_pred - y
        return (2.0 / len(y)) * (X.T @ err)

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        # Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        # Необходима в связи c наличием интерфейса analytic_solution у любого лосса; 
        # self-injection даёт возможность выбирать, какое именно closed-form решение использовать
        return self.analytic_solution_func(X, y)
        
    
    @classmethod
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи классического аналитического решения
        """
        return np.linalg.inv(X.T @ X) @ X.T @ y
    
    @classmethod
    def _svd_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи аналитического решения на SVD
        """
        r = np.linalg.matrix_rank(X)
        if r == 0:
            return np.zeros(X.shape[1])
        if r >= min(X.shape):
            return np.linalg.lstsq(X, y, rcond=None)[0]

        from scipy.sparse.linalg import svds

        U, S, Vt = svds(X, k=r, solver='arpack', tol=0)

        order = np.argsort(S)[::-1]
        S = S[order]
        U = U[:, order]
        Vt = Vt[order, :]

        return Vt.T @ (np.diag(1.0 / S) @ (U.T @ y))


class L2Regularization(LossFunction):

    def __init__(self, core_loss: LossFunction, mu_rate: float = 1.0,
                 analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        self.core_loss = core_loss
        self.mu_rate = mu_rate

        # analytic_solution_func is meant to be passed separately, 
        # as it is not linear to core solution

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        core_value = self.core_loss.loss(X, y, w)
        penalty_value = 0.5 * self.mu_rate * float(np.sum(w ** 2))
        return float(core_value + penalty_value)
    

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:

        core_part = self.core_loss.gradient(X, y, w)

        penalty_part = self.mu_rate * w

        return core_part + penalty_part


class LogCoshLoss(LossFunction):
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        residual = X @ w - y
        values = np.logaddexp(residual, -residual) - np.log(2.0)
        return float(np.mean(values))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        residual = X @ w - y
        return (1.0 / len(y)) * (X.T @ np.tanh(residual))


class HuberLoss(LossFunction):
    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        residual = X @ w - y
        abs_residual = np.abs(residual)
        quadratic = 0.5 * residual ** 2
        linear = self.delta * abs_residual - 0.5 * (self.delta ** 2)
        values = np.where(abs_residual < self.delta, quadratic, linear)
        return float(np.mean(values))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        residual = X @ w - y
        abs_residual = np.abs(residual)
        grad_residual = np.where(abs_residual < self.delta, residual, self.delta * np.sign(residual))
        return (1.0 / len(y)) * (X.T @ grad_residual)



class CustomLinearRegression(LinearRegressionInterface):
    def __init__(
        self,
        optimizer: AbstractOptimizer,
        # l2_coef: float = 0.0,
        loss_function: LossFunction = MSELoss()
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)

        # self.l2_coef = l2_coef
        self.loss_function = loss_function
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        r"""
        returns: np.ndarray, вектор \hat{y}
        """
        if self.w is None:
            raise ValueError("Model is not fitted")
        return X @ self.w

    def compute_gradients(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> np.ndarray:
        """
        returns: np.ndarray, градиент функции потерь при текущих весах (self.w)
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if self.w is None:
            raise ValueError("Model is not fitted")
        X_use = self.X_train if X_batch is None else X_batch
        y_use = self.y_train if y_batch is None else y_batch
        return self.loss_function.gradient(X_use, y_use, self.w)


    def compute_loss(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> float:
        """
        returns: np.ndarray, значение функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if self.w is None:
            raise ValueError("Model is not fitted")
        X_use = self.X_train if X_batch is None else X_batch
        y_use = self.y_train if y_batch is None else y_batch
        return self.loss_function.loss(X_use, y_use, self.w)


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.
        
        X: np.ndarray, 
        y: np.ndarray
        """
        # TODO: реализовать обучение модели
        self.X_train, self.y_train = X, y

        if self.w is None:
            self.w = np.zeros(X.shape[1])

        self.optimizer.optimize()
