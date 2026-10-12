"""SLM Week 3 on the real Palo Alto dataset: OLS + gradient descent + MLflow."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .palo_alto_schema import BASE_FEATURES, REGRESSION_TARGET

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/chargeup_sessions_clean.csv'
METRICS=ROOT/'reports/week03_linear_regression_metrics.json'
PRED=ROOT/'reports/week03_test_predictions.csv'
TIME_FEATURES=['hour','day_of_week']
FEATURES=BASE_FEATURES+TIME_FEATURES

class GradientDescentLinearRegressor:
    def __init__(self,learning_rate=.03,max_iter=5000,tol=1e-10):
        self.learning_rate=learning_rate; self.max_iter=max_iter; self.tol=tol
        self.x_mean_=self.x_scale_=self.coef_=None; self.y_mean_=self.y_scale_=self.intercept_=None
        self.n_iter_=0; self.final_loss_=None; self.loss_history_=[]
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float).reshape(-1)
        self.x_mean_=X.mean(0); self.x_scale_=X.std(0); self.x_scale_[self.x_scale_==0]=1
        self.y_mean_=float(y.mean()); ys=float(y.std()); self.y_scale_=ys if ys>0 else 1.0
        Xs=(X-self.x_mean_)/self.x_scale_; yt=(y-self.y_mean_)/self.y_scale_
        w=np.zeros(Xs.shape[1]); b=0.; prev=np.inf; n=len(Xs)
        for i in range(1,self.max_iter+1):
            p=Xs@w+b; e=p-yt; loss=float(np.mean(e*e)); w-=self.learning_rate*(2/n)*(Xs.T@e); b-=self.learning_rate*float((2/n)*e.sum())
            if i==1 or i%100==0: self.loss_history_.append({'iteration':i,'mse_scaled':loss})
            if abs(prev-loss)<self.tol: self.n_iter_=i; prev=loss; break
            prev=loss
        else: self.n_iter_=self.max_iter
        self.coef_=w; self.intercept_=b; self.final_loss_=float(prev); return self
    def predict(self,X):
        X=np.asarray(X,float); xs=(X-self.x_mean_)/self.x_scale_; return (xs@self.coef_+self.intercept_)*self.y_scale_+self.y_mean_

def load_frame(path=DATA):
    df=pd.read_csv(path,parse_dates=['timestamp']).sort_values(['timestamp','station_id']).reset_index(drop=True)
    h=df.timestamp.dt.hour; d=df.timestamp.dt.dayofweek; df['hour']=h.astype(float); df['day_of_week']=d.astype(float)
    df[FEATURES]=df[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0)
    return df

def split(df,test_fraction=.2):
    times=np.array(sorted(df.timestamp.unique())); cut=pd.Timestamp(times[int(len(times)*(1-test_fraction))])
    tr=df[df.timestamp<cut-pd.Timedelta(hours=1)].copy(); te=df[df.timestamp>=cut].copy()
    return tr,te,cut

def metrics(y,p): return {'mae':float(mean_absolute_error(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'r2':float(r2_score(y,p))}

def log_mlflow(out,metrics_path,pred_path):
    try:
        import mlflow
        mlflow.set_experiment('ChargeUp-PaloAlto-Week03')
        with mlflow.start_run(run_name='real-data-ols-vs-gradient-descent') as run:
            mlflow.log_param('dataset','City of Palo Alto ChargePoint CY20Q4'); mlflow.log_param('feature_count',len(FEATURES)); mlflow.log_param('split','chronological_with_1h_purge')
            for name in ['ols','gradient_descent']:
                for key in ['mae','rmse','r2']: mlflow.log_metric(f'{name}_{key}',float(out[name][key]))
            mlflow.log_artifact(str(metrics_path)); mlflow.log_artifact(str(pred_path)); return {'logged':True,'run_id':run.info.run_id}
    except Exception as exc: return {'logged':False,'reason':str(exc)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=DATA); ap.add_argument('--no-mlflow',action='store_true'); args=ap.parse_args()
    df=load_frame(args.input); tr,te,cut=split(df); Xtr=tr[FEATURES].to_numpy(float); Xte=te[FEATURES].to_numpy(float); ytr=tr[REGRESSION_TARGET].to_numpy(float); yte=te[REGRESSION_TARGET].to_numpy(float)
    ols=Pipeline([('scale',StandardScaler()),('model',LinearRegression())]); ols.fit(Xtr,ytr); op=ols.predict(Xte)
    gd=GradientDescentLinearRegressor(); gd.fit(Xtr,ytr); gp=gd.predict(Xte)
    out={'week':3,'dataset':'City of Palo Alto ChargePoint CY20Q4','target':REGRESSION_TARGET,'features':FEATURES,'split_time':cut.isoformat(),'train_rows':len(tr),'test_rows':len(te),'ols':metrics(yte,op),'gradient_descent':{**metrics(yte,gp),'iterations_run':gd.n_iter_,'final_scaled_mse':gd.final_loss_,'loss_history':gd.loss_history_}}
    pred=te[['station_id','timestamp',REGRESSION_TARGET]].copy(); pred['ols_prediction_kwh']=op; pred['gradient_descent_prediction_kwh']=gp
    METRICS.parent.mkdir(parents=True,exist_ok=True); pred.to_csv(PRED,index=False); METRICS.write_text(json.dumps(out,indent=2),encoding='utf-8')
    out['mlflow']={'logged':False,'reason':'disabled'} if args.no_mlflow else log_mlflow(out,METRICS,PRED); METRICS.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
