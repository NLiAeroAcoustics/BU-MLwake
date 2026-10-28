import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error


class post:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.test_pred = None

    def _predict(self):
        if self.test_pred is None:
            self.test_pred = self.model.predict(self.x_test)
        return self.test_pred

    def pixelError(self):
        pred = self._predict()
        mae = mean_absolute_error(self.y_test.reshape(-1), pred.reshape(-1))
        r2 = r2_score(self.y_test.reshape(-1), pred.reshape(-1))
        print('mean mae on test data: ', mae)
        print('R-squared on test data: ', r2)
        return mae, r2

    def pixelError_mc(self):
        pred = self._predict()
        n_channels = self.y_test.shape[-1]

        mae = []
        r2 = []
        for i in range(n_channels):
            y_true_i = self.y_test[..., i].reshape(-1)
            y_pred_i = pred[..., i].reshape(-1)
            mae.append(mean_absolute_error(y_true_i, y_pred_i))
            r2.append(r2_score(y_true_i, y_pred_i))

        print('mean mae on test data: ', mae)
        print('R-squared on test data: ', r2)
        return mae, r2

    def circError(self):
        pred = self._predict()
        y_true_avg = self.y_test.mean(axis=1)
        y_pred_avg = pred[..., 0].mean(axis=1)
        mae = mean_absolute_error(y_true_avg, y_pred_avg)
        r2 = r2_score(y_true_avg, y_pred_avg)
        print('mean mae on test data: ', mae)
        print('R-squared on test data: ', r2)
        return mae, r2

    def circError_mc(self):
        pred = self._predict()
        n_channels = self.y_test.shape[-1]

        mae = []
        r2 = []
        for i in range(n_channels):
            y_true_i = self.y_test[..., i].mean(axis=1).reshape(-1)
            y_pred_i = pred[..., i].mean(axis=1).reshape(-1)
            mae.append(mean_absolute_error(y_true_i, y_pred_i))
            r2.append(r2_score(y_true_i, y_pred_i))

        print('mean mae on test data: ', mae)
        print('R-squared on test data: ', r2)
        return mae, r2

    def Error_pr_tr(self, gamma=1.4):
        pred = self._predict()

        pr_true = self.y_test[:, 0]
        tr_true = self.y_test[:, 1]
        pr_pred = pred[:, 0]
        tr_pred = pred[:, 1]

        exp = (gamma - 1.0) / gamma
        eta_true = (np.power(pr_true, exp) - 1.0) / (tr_true - 1.0)
        eta_pred = (np.power(pr_pred, exp) - 1.0) / (tr_pred - 1.0)

        mae_pr = mean_absolute_error(pr_true, pr_pred)
        mae_tr = mean_absolute_error(tr_true, tr_pred)
        mae_eta = mean_absolute_error(eta_true, eta_pred)

        r2_pr = r2_score(pr_true, pr_pred)
        r2_tr = r2_score(tr_true, tr_pred)
        r2_eta = r2_score(eta_true, eta_pred)

        print('Pressure ratio mae on test data: ', mae_pr)
        print('Pressure ratio R-squared on test data: ', r2_pr)
        print('Temperature ratio mae on test data: ', mae_tr)
        print('Temperature ratio R-squared on test data: ', r2_tr)
        print('Isentropic efficiency mae on test data: ', mae_eta)
        print('Isentropic efficiency R-squared on test data: ', r2_eta)

        return mae_pr, mae_tr, mae_eta, r2_pr, r2_tr, r2_eta

    def Error_scalar(self):
        pred = self._predict()
        mae = mean_absolute_error(self.y_test, pred)
        r2 = r2_score(self.y_test, pred)
        print('mae on test data: ', mae)
        print('R-squared on test data: ', r2)
        return mae, r2
