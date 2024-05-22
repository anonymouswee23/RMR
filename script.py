import portpy.photon as pp
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import algorithms
import pickle

def l2_norm(matrix):
    values = np.linalg.eig(np.transpose(matrix) @ matrix)[0]
    return math.sqrt(np.max(np.abs(values)))

def relative_error(dose_sprs, dose_Full):
    return (np.linalg.norm(dose_Full - dose_sprs) / np.linalg.norm(dose_Full)) * 100

def search1(name, matrix, percentages, nnz):
    thresholds = []
    prev_threshold = 0.1
    func = getattr(algorithms, name)
    nonzeros = func(matrix, prev_threshold)
    for percentage in percentages:
        tar_nonzeros = percentage * nnz / 100
        threshold = prev_threshold
        count = 0
        while np.abs(1 - nonzeros / tar_nonzeros) > 0.025 and count < 20:
            threshold *= nonzeros / tar_nonzeros
            nonzeros = func(matrix, threshold)
            count += 1
        prev_threshold = threshold
        thresholds.append(threshold)
    return thresholds

def search2(name, matrix, percentages, nnz):
    thresholds = []
    func = getattr(algorithms, name)
    for percentage in percentages:
        tar_nonzeros = percentage * nnz / 100
        threshold = tar_nonzeros
        nonzeros = func(matrix, threshold)
        count = 0
        while np.abs(1 - tar_nonzeros / nonzeros) > 0.025 and count < 20:
            threshold *= tar_nonzeros / nonzeros
            nonzeros = func(matrix, threshold)
            count += 1
        thresholds.append(threshold)
    return thresholds

def search3(name, matrix, percentages, nnz):
    thresholds = []
    prev_threshold = 250
    func = getattr(algorithms, name)
    tick = time.time()
    nonzeros = func(matrix, prev_threshold)
    runtime = time.time() - tick
    min_threshold = prev_threshold / math.sqrt(600 / runtime)
    prev_threshold = 500
    nonzeros = func(matrix, prev_threshold)
    for t, percentage in enumerate(percentages):
        tar_nonzeros = percentage * nnz / 100
        threshold = prev_threshold
        count = 0
        while np.abs(1 - nonzeros / tar_nonzeros) > 0.025 and count < 20:
            threshold *= nonzeros / tar_nonzeros
            if threshold < min_threshold and t == 1:
                threshold = min_threshold
                count = 20
            elif threshold < min_threshold and t == 0:
                return False
            else:
                nonzeros = func(matrix, threshold)
            count += 1
        prev_threshold = threshold
        thresholds.append(threshold)
    return thresholds

def run_algorithm(i, func1, func2, matrix, nnz, percentages, total_points, repetitions):
    print("Starting", func1)
    if func1 == "AKL13":
        thresholds0 = search2(func2, matrix, percentages, nnz)
    elif func1 == "DZ11":
        thresholds0 = search3(func2, matrix, percentages, nnz)
    else:
        thresholds0 = search1(func2, matrix, percentages, nnz)
    if thresholds0 == False:
        for key in dict[func1]:
            if key != "Thresholds":
                dict[func1][key].extend([0] * (total_points * repetitions))
            else:
                dict[func1][key].extend([0] * 2)
        return
    dict[func1]["Thresholds"].extend(thresholds0)
    print(f"{func1} thresholds: {thresholds0}")
    all_thresholds = np.linspace(thresholds0[0], thresholds0[1], num=total_points)
    for j, threshold in enumerate(all_thresholds):
        for k in range(repetitions):
            print(f"Patient {i}, Algorithm {func1}, Point {j}, Repetition {k}")
            func = getattr(algorithms, func1)
            tick = time.time()
            B = func(matrix, threshold)
            tock = time.time()
            dict[func1]["Times"].append(tock - tick)
            print("Time:", tock - tick)
            B_nonzeros = len(B.nonzero()[0])
            dict[func1]["Nonzeros"].append(B_nonzeros)
            print("Number of nonzeros of B:", B_nonzeros)
            AB_norm = l2_norm(matrix - B)
            dict[func1]["L2 norms"].append(AB_norm)
            print("L2 norm of A-B:", AB_norm)
            inf_matrix.A = B
            plan1 = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria)
            opt1 = pp.Optimization(plan1, opt_params=opt_params)
            opt1.create_cvxpy_problem()
            tick = time.time()
            x_B = opt1.solve(solver='MOSEK', verbose=False)
            tock = time.time()
            dict[func1]["Optimization times"].append(tock - tick)

            opt.vars['x'].value = x_B['optimal_intensity']
            violation = 0
            for constraint in opt.constraints[2:]:
                violation += np.sum(constraint.violation())
            dict[func1]["Feasibility violations"].append(violation)
            print("Feasibility violation:", violation)
            objective_function_value(x_B['optimal_intensity'], func1)
            dose_1d = B @ (x_B['optimal_intensity'] * plan1.get_num_of_fractions())
            dose_full = matrix @ (x_B['optimal_intensity'] * plan1.get_num_of_fractions())
            np.save(f"Vectors/{func1}_{i}_{j}_{k}_x", x_B['optimal_intensity'])
            np.save(f"Vectors/{func1}_{i}_{j}_{k}_Bx", dose_1d)
            np.save(f"Vectors/{func1}_{i}_{j}_{k}_Ax", dose_full)
            dict[func1]["Dose discrepancy"].append(relative_error(dose_1d, dose_full))
	    tock = time.time()
	    dict["Total time"] = tock - main_tick
            with open('output.pkl', 'wb') as file:
                pickle.dump(dict, file)

def objective_function_value(x, name):
    obj_funcs = opt_params['objective_functions'] if 'objective_functions' in opt_params else []
    obj = 0
    for i in range(len(obj_funcs)):
        if obj_funcs[i]['type'] == 'quadratic-overdose':
            if obj_funcs[i]['structure_name'] in opt.my_plan.structures.get_structures():
                struct = obj_funcs[i]['structure_name']
                if len(inf_matrix_full.get_opt_voxels_idx(struct)) == 0:  # check if there are any opt voxels for the structure
                    continue
                dose_gy = opt.get_num(obj_funcs[i]['dose_gy']) / clinical_criteria.get_num_of_fractions()
                dO = np.maximum(A[inf_matrix_full.get_opt_voxels_idx(struct), :] @ x - dose_gy, 0)
                obj += (1 / len(inf_matrix_full.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * np.sum(dO ** 2))
        elif obj_funcs[i]['type'] == 'quadratic-underdose':
            if obj_funcs[i]['structure_name'] in opt.my_plan.structures.get_structures():
                struct = obj_funcs[i]['structure_name']
                if len(inf_matrix_full.get_opt_voxels_idx(struct)) == 0:
                    continue
                dose_gy = opt.get_num(obj_funcs[i]['dose_gy']) / clinical_criteria.get_num_of_fractions()
                dU = np.minimum(A[inf_matrix_full.get_opt_voxels_idx(struct), :] @ x - dose_gy, 0)
                obj += (1 / len(inf_matrix_full.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * np.sum(dU ** 2))
        elif obj_funcs[i]['type'] == 'quadratic':
            if obj_funcs[i]['structure_name'] in opt.my_plan.structures.get_structures():
                struct = obj_funcs[i]['structure_name']
                if len(inf_matrix_full.get_opt_voxels_idx(struct)) == 0:
                    continue
                obj += (1 / len(inf_matrix_full.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * np.sum((A[inf_matrix_full.get_opt_voxels_idx(struct), :] @ x) ** 2))
        elif obj_funcs[i]['type'] == 'smoothness-quadratic':
            [Qx, Qy, num_rows, num_cols] = opt.get_smoothness_matrix(inf_matrix.beamlets_dict)
            smoothness_X_weight = 0.6
            smoothness_Y_weight = 0.4
            obj += obj_funcs[i]['weight'] * (smoothness_X_weight * (1 / num_cols) * np.sum((Qx @ x) ** 2) +
                                                    smoothness_Y_weight * (1 / num_rows) * np.sum((Qy @ x) ** 2))
    dict[name]["Objective values"].append(obj)
    print("Objective function value:", obj)


main_tick = time.time()
total_points = 35
list_of_algorithms = ["Naive", "AHK06", "AKL13", "RMR", "DZ11"]
total_repetitions = [1, 5, 5, 5, 5]
data_dir = r'data'
data = pp.DataExplorer(data_dir=data_dir)

dict = {"Index": [], "Nonzeros": [], "Shapes": [], "L2 norms": []}
for name in list_of_algorithms:
    dict[name] = {"Nonzeros": [], "L2 norms": [], "Times": [], "Dose discrepancy": [], "Optimization times": [], "Objective values": [], "Feasibility violations": [], "Thresholds": []}
    
patients_list = np.arange(2, 51)
np.random.shuffle(patients_list)

for i in patients_list:
    dict["Index"].append(i)
    print("Starting Patient", i)
    data.patient_id = 'Lung_Patient_' + str(i)
    ct = pp.CT(data)
    structs = pp.Structures(data)
    beams = pp.Beams(data)
    # Pick a protocol
    protocol_name = 'Lung_2Gy_30Fx'
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)

    opt_params = data.load_config_opt_params(protocol_name=protocol_name)
    structs.create_opt_structures(opt_params=opt_params)
    beams_full = pp.Beams(data, load_inf_matrix_full=True)
    inf_matrix_full = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams_full, is_full=True)
    plan_full = pp.Plan(ct, structs, beams, inf_matrix_full, clinical_criteria)
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    opt = pp.Optimization(plan_full, opt_params=opt_params)
    opt.create_cvxpy_problem()

    A = inf_matrix_full.A
    A_nonzeros = len(A.nonzero()[0])
    dict["Nonzeros"].append(A_nonzeros)
    print("Number of nonzeros of A:", A_nonzeros)
    dict["Shapes"].append(A.shape)
    A_norm = l2_norm(A)
    dict["L2 norms"].append(A_norm)
    print("L2 norm of A:", A_norm)

    sparsification_percentages = np.array([1, 5])
    for j in range(len(list_of_algorithms)):
        run_algorithm(i, list_of_algorithms[j], f"{list_of_algorithms[j]}_nonzeros", A, A_nonzeros, sparsification_percentages, total_points, total_repetitions[j])

tock = time.time()
dict["Total time"] = tock - main_tick
print("Total time:", tock - main_tick)
with open('output.pkl', 'wb') as file:
    pickle.dump(dict, file)
