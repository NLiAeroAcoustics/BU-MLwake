from scipy.io import savemat, loadmat
import pickle 
def saveasmat_mc(Dir, model,x_train,x_test,y_train,y_test,label_train,coordinate_train,label_test,coordinate_test, nfile = None):       

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
    
    tke_train = outputs_train[:, :, :, 0]
    ls_train = outputs_train[:, :, :, 1]
    vx_train = outputs_train[:, :, :, 2]
    vt_train = outputs_train[:, :, :, 3]
    p_train = outputs_train[:, :, :, 4]
    T_train = outputs_train[:, :, :, 5]
    vm_train = outputs_train[:, :, :, 6]
  
    
    train_prediction = dict(inputs = inputs_train, 
                            label = label_train,
                            x = gx_train,
                            y = gy_train,
                            z = gz_train,
                            r = gr_train, 
                            th = gth_train,
                            tke_train = outputs_train[:, :, :, 0],
                            ls_train = outputs_train[:, :, :, 1],
                            vx_train = outputs_train[:, :, :, 2],
                            vt_train = outputs_train[:, :, :, 3],
                            p_train = outputs_train[:, :, :, 4],
                            T_train = outputs_train[:, :, :, 5],
                            vm_train = outputs_train[:, :, :, 6],
                            tke_train_pred = outputs_train_pred[:, :, :, 0],
                            ls_train_pred = outputs_train_pred[:, :, :, 1],
                            vx_train_pred = outputs_train_pred[:, :, :, 2],
                            vt_train_pred = outputs_train_pred[:, :, :, 3],
                            p_train_pred = outputs_train_pred[:, :, :, 4],
                            T_train_pred = outputs_train_pred[:, :, :, 5],
                            vm_train_pred = outputs_train_pred[:, :, :, 6])
    
    test_prediction = dict(inputs = inputs_test, 
                           label = label_test,
                           x = gx_test,
                           y = gy_test,
                           z= gz_test,
                           r = gr_test, 
                           th = gth_test,
                           tke_test = outputs_test[:, :, :, 0],
                           ls_test = outputs_test[:, :, :, 1],
                           vx_test = outputs_test[:, :, :, 2],
                           vt_test = outputs_test[:, :, :, 3],
                           p_test = outputs_test[:, :, :, 4],
                           T_test = outputs_test[:, :, :, 5],
                           vm_test = outputs_test[:, :, :, 6],
                           tke_test_pred = outputs_test_pred[:, :, :, 0],
                           ls_test_pred = outputs_test_pred[:, :, :, 1],
                           vx_test_pred = outputs_test_pred[:, :, :, 2],
                           vt_test_pred = outputs_test_pred[:, :, :, 3],
                           p_test_pred = outputs_test_pred[:, :, :, 4],
                           T_test_pred = outputs_test_pred[:, :, :, 5],
                           vm_test_pred = outputs_test_pred[:, :, :, 6],)

    
    savemat(Dir+"/train_prediction.mat", train_prediction)
    savemat(Dir+"/test_prediction.mat", test_prediction)
    
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