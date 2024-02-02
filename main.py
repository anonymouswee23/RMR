import portpy.photon as pp
import algorithms
import numpy as np
import math
import matplotlib.pyplot as plt

def l2_norm(matrix):
    values, vectors = np.linalg.eig(np.transpose(matrix) @ matrix)
    return math.sqrt(np.max(np.abs(values)))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--method', type=str, choices=['Naive', 'AHK06', 'AKL13', 'DZ11', 'RMR'], help='The name of method.'
    )
    parser.add_argument(
        '--patient', type=str, help='Patient\'s name'
    )
    parser.add_argument(
        '--threshold', type=float, help='The threshold using for the input of algorithm.'
    )
    parser.add_argument(
        '--solver', type=str, default='SCS', help='The name of solver for solving the optimization problem'
    )

    args = parser.parse_args()
    # Use PortPy DataExplorer class to explore PortPy data
    data = pp.DataExplorer(data_dir='')
    # Pick a patient
    data.patient_id = args.patient
    # Load ct, structure set, beams for the above patient using CT, Structures, and Beams classes
    ct = pp.CT(data)
    structs = pp.Structures(data)
    beams = pp.Beams(data)
    # Pick a protocol
    protocol_name = 'Lung_2Gy_30Fx'
    # Load clinical criteria for a specified protocol
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)
    # Load hyper-parameter values for optimization problem for a specified protocol
    opt_params = data.load_config_opt_params(protocol_name=protocol_name)
    # Create optimization structures (i.e., Rinds)
    structs.create_opt_structures(opt_params=opt_params)
    # create plan_full object by specifying load_inf_matrix_full=True
    beams_full = pp.Beams(data, load_inf_matrix_full=True)
    # load influence matrix based upon beams and structure set
    inf_matrix_full = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams_full, is_full=True)
    plan_full = pp.Plan(ct, structs, beams, inf_matrix_full, clinical_criteria)
    # Load influence matrix
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    A = inf_matrix_full.A
    print("number of non-zeros of the original matrix: ", len(A.nonzero()[0]))
    
    method = getattr(algorithms, args.method)
    B = method(A, args.threshold)
    print("number of non-zeros of the sparsed matrix: ", len(B.nonzero()[0]))
    print("relative L2 norm (%): ", l2_norm(A - B) / l2_norm(A) * 100)

    inf_matrix.A = B
    plan = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria)
    opt = pp.Optimization(plan, opt_params=opt_params)
    opt.create_cvxpy_problem()
    x = opt.solve(solver=args.solver, verbose=False)

    dose_1d = B @ (x['optimal_intensity'] * plan.get_num_of_fractions())
    dose_full = A @ (x['optimal_intensity'] * plan.get_num_of_fractions())
    print("relative dose discrepancy (%): ", (np.linalg.norm(dose_full - dose_1d) / np.linalg.norm(dose_full)) * 100)

    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    fig, ax = plt.subplots(figsize=(12, 8))
    # Turn on norm flag for same normalization for sparse and full dose.
    ax = pp.Visualization.plot_dvh(plan, dose_1d=dose_1d , struct_names=struct_names, style='solid', ax=ax, norm_flag=True)
    ax = pp.Visualization.plot_dvh(plan_full, dose_1d=dose_full, struct_names=struct_names, style='dotted', ax=ax, norm_flag=True)
    plt.savefig(str(args.method) + "_" + str(args.threshold) + "_" + str(args.patient) + ".pdf")
