from scipy.io import savemat, loadmat
import pickle 
import os
def saveasmat_mc(Dir, model,x_train,x_test,y_train,y_test,label_train,coordinate_train,label_test,coordinate_test, nfile = None, output_keys = None):       

    gx_train = coordinate_train[:, :, :, 0]
    gy_train = coordinate_train[:, :, :, 1]
    gz_train = coordinate_train[:, :, :, 2]
    gr_train = coordinate_train[:, :, :, 3]
    gth_train = coordinate_train[:, :, :, 4]
    
    gx_test = coordinate_test[:, :, :, 0]
    gy_test = coordinate_test[:, :, :, 1]
    gz_test = coordinate_test[:, :, :, 2]
    gr_test = coordinate_test[:, :, :, 3]
    gth_test = coordinate_test[:, :, :, 4]    
    
    with open(nfile, 'rb') as n:
        norm = pickle.load(n)
        input_mean = norm['input_mean']
        input_std = norm['input_std']
        output_mean = norm['output_mean']
        output_std = norm['output_std']
        epsi = norm['epsi']
    
    prediction_train = model.predict(x_train)
    prediction_test  = model.predict(x_test)
    
    inputs_train = x_train*(input_std+epsi)+input_mean
    inputs_test = x_test*(input_std+epsi)+input_mean
    
    outputs_train = y_train*(output_std+epsi)+output_mean
    outputs_test = y_test*(output_std+epsi)+output_mean  
    
    outputs_train_pred = prediction_train*(output_std+epsi)+output_mean
    outputs_test_pred = prediction_test*(output_std+epsi)+output_mean    
    
    # Default names preserve the existing seven-channel MAT fields.
    if output_keys is None:
        output_keys = ["tke", "LenScale", "vx", "vt", "p", "T", "v_mag"]
    if len(output_keys) != y_train.shape[-1] or len(output_keys) != y_test.shape[-1]:
        raise ValueError("Pass output_keys in the same order as the training output channels.")
    names = [{"LenScale": "ls", "v_mag": "vm"}.get(key, key) for key in output_keys]
    if len(set(names)) != len(names):
        raise ValueError("output_keys must have unique MAT field names.")

    train_prediction = dict(inputs=inputs_train, label=label_train,
                            x=gx_train, y=gy_train, z=gz_train,
                            r=gr_train, th=gth_train)
    test_prediction = dict(inputs=inputs_test, label=label_test,
                           x=gx_test, y=gy_test, z=gz_test,
                           r=gr_test, th=gth_test)
    for i, name in enumerate(names):
        train_prediction[name + "_train"] = outputs_train[:, :, :, i]
        train_prediction[name + "_train_pred"] = outputs_train_pred[:, :, :, i]
        test_prediction[name + "_test"] = outputs_test[:, :, :, i]
        test_prediction[name + "_test_pred"] = outputs_test_pred[:, :, :, i]

    os.makedirs(Dir, exist_ok=True)
    savemat(Dir + "/train_prediction.mat", train_prediction)
    savemat(Dir + "/test_prediction.mat", test_prediction)
    

def pred2mat_mc(varname, Dir, model,x_test,label_test,coordinate_test):       
    with open(nfile, 'rb') as n:
        norm = pickle.load(n)
        input_mean = norm['input_mean']
        input_std = norm['input_std']
        output_mean = norm['output_mean']
        output_std = norm['output_std']
        epsi = norm['epsi']
        
    gx_test = coordinate_test[:, :, :, 0]
    gy_test = coordinate_test[:, :, :, 1]
    gz_test = coordinate_test[:, :, :, 2]
    gr_test = coordinate_test[:, :, :, 3]
    gth_test = coordinate_test[:, :, :, 4] 
    
    prediction_test  = model.predict(x_test)
    
    inputs_test = x_test*(input_std+epsi)+input_mean
    outputs_test_pred = prediction_test*(output_std+epsi)+output_mean  
    
    test_prediction = dict(inputs = inputs_test, 
                           label = label_test,
                           x = gx_test,
                           y = gy_test,
                           z= gz_test,
                           r = gr_test, 
                           th = gth_test,
                           tke_test_pred = outputs_test_pred[:, :, :, 0],
                           vx_test_pred = outputs_test_pred[:, :, :, 3],
                           vt_test_pred = outputs_test_pred[:, :, :, 4],
                           vm_test_pred = outputs_test_pred[:, :, :, 7],
                           p_test_pred = outputs_test_pred[:, :, :, 5],
                           T_test_pred = outputs_test_pred[:, :, :, 6],
                           ls_test_pred = outputs_test_pred[:, :, :, 2])
    savemat(Dir+"/test_prediction.mat", test_prediction)    