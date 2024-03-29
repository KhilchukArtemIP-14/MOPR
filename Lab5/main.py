import numpy as np
import pandas as pd
import warnings

def get_wq_norm_sum(matrix):
    wQ = np.sum(matrix, axis=1).reshape(-1, 1)
    #print(f"wQ:\n{wQ}")
    #print(f"sum: {np.sum(wQ)}")
    wQ_norm = wQ/np.sum(wQ)
    #print(f"wQ_norm:\n{wQ_norm}")
    return wQ_norm

def get_wq_norm_geometric(matrix):
    degree = matrix.shape[1]
    #for row in matrix:
    #    formatted_row = '*'.join('{:.2f}'.format(member).rstrip('0').rstrip('.') for member in row)
    #    print(formatted_row)
    wQ = np.prod(matrix, axis=1).reshape(-1, 1)
    wQ = np.power(wQ, 1 / degree)
    #print(wQ)
    wQ_norm = wQ/np.sum(wQ)
    #print(np.sum(wQ))
    #print(wQ_norm)
    return wQ_norm

def calculate_alternatives_weight(criterion_weights, alternative_comparisons_list, get_wQ_method):
    wQ_norm = get_wQ_method(criterion_weights)
    print(wQ_norm)

    wQx_dict = dict()
    for index, criterion in enumerate(alternative_comparisons_list):
        wQx_dict[f"wQ{index+1}_norm"] = get_wQ_method(criterion)
    print(wQx_dict)
    result_matrix = np.concatenate(tuple(wQx_dict.values()), axis=1)
    print(result_matrix)

    final_result = np.dot(result_matrix, wQ_norm)
    return final_result

def get_eigenvaluemax_approx(matrix):
    column_sums = np.sum(matrix, axis=0)
    print(f"Column sums: {column_sums}")
    wQ_norm_geom = get_wq_norm_geometric(matrix)
    print(f"wQ_norm:{wQ_norm_geom}")

    eigenvalue_max= np.dot(column_sums,wQ_norm_geom)
    print(f"eigenValue approx max: {eigenvalue_max[0]}")
    return eigenvalue_max[0]

def get_eighenvaluemax_precise(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # eigenvectors are normalized by default, thus creating slight deviation from what wolfram says
    sort_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_eigenvectors = eigenvectors[sort_indices, :]

    min_rowsum =  np.min(np.sum(matrix, axis=1))
    max_rowsum =  np.max(np.sum(matrix, axis=1))

    for index, value in enumerate(eigenvalues):
        if(value>0 and abs(value.imag)==0):
            if(np.count_nonzero(eigenvalues == value)==1):
                if(np.all(eigenvectors[:, index] > 0)):
                    if(value<max_rowsum and value>min_rowsum):

                        return value.real;
    raise Exception("No eigenvalue")

def asses_consistency_of_hierarchy(criterion_weights, alternative_comparisons_list, get_wQ_method):

        IVU_dict={
            1:0,        2:0,  3:0.52,    4:0.89,
            5:1.12,  6:1.24,  7:1.32,    8:1.41,
            9:1.45, 10:1.49, 11:1.51,   12:1.48
        }

        uzgodzhenist = pd.DataFrame(columns=['Matrix','Dimension','EigenvalueMax','IVU'])

        new_row={'Matrix':'Q',
                 'Dimension':criterion_weights.shape[1],
                 'EigenvalueMax':get_eigenvaluemax_approx(criterion_weights),
                 'IVU':IVU_dict[criterion_weights.shape[1]]}
        uzgodzhenist.loc[0]=new_row

        for index, criterion in enumerate(alternative_comparisons_list):
            new_row = {'Matrix': f'Q{index+1}',
                       'Dimension': criterion.shape[1],
                       'EigenvalueMax': get_eigenvaluemax_approx(criterion),
                       'IVU': IVU_dict[criterion.shape[1]]}
            uzgodzhenist.loc[len(uzgodzhenist)] = new_row
        uzgodzhenist = uzgodzhenist.set_index('Matrix')

        uzgodzhenist["IU"] = (uzgodzhenist["EigenvalueMax"] - uzgodzhenist["Dimension"])/(uzgodzhenist["Dimension"] - 1)
        uzgodzhenist["OU"] = uzgodzhenist["IU"] / uzgodzhenist["IVU"]
        print(uzgodzhenist)

        wQ_norm = get_wQ_method(criterion_weights)
        IUI = uzgodzhenist.loc['Q', 'IU'] + np.dot(wQ_norm.T, uzgodzhenist.loc['Q1':'Q4', 'IU'].values.reshape(-1, 1))[0, 0]
        IVUI = uzgodzhenist.loc['Q', 'IVU'] + np.dot(wQ_norm.T, uzgodzhenist.loc['Q1':'Q4', 'IVU'].values.reshape(-1, 1))[0, 0]
        print(f"IUI: {IUI}; IVUI: {IVUI}")
        OU = IUI/IVUI
        return OU

def get_crit_values(alt_labels, crit_labels, criterion_weights, alternative_comparisons_list, get_wQ_method):
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide")

    final_result_df = pd.DataFrame({
        'Label': alt_labels,
        'GlobalWeight': calculate_alternatives_weight(criterion_weights,
                                                      alternative_comparisons_list,
                                                      get_wQ_method)
                                                      .flatten()
    })
    for index,crit_label in enumerate(crit_labels):
        final_result_df[crit_label]=get_wQ_method(alternative_comparisons_list[index]).flatten()


    final_result_df = final_result_df.sort_values(by='GlobalWeight', ascending=False)
    final_result_df = final_result_df.reset_index(drop=True)

    print(final_result_df)

    global_criterions = pd.DataFrame({
        'Label': crit_labels,
        'Value': get_wQ_method(criterion_weights).flatten(),
    })
    global_criterions = global_criterions.set_index("Label")
    print(global_criterions)

    crit_values = pd.DataFrame(columns=["AlternativePairs", *crit_labels])

    for i in range(crit_values.shape[1]-1):
        for j in range(i + 1, crit_values.shape[1]):
            row_i = final_result_df.loc[i]
            row_j = final_result_df.loc[j]

            new_row = {"AlternativePairs": f"({i + 1},{j + 1})"}
            for criterion in crit_labels:
                value = (row_j["GlobalWeight"] - row_i["GlobalWeight"]) / (row_j[criterion] - row_i[criterion]) /global_criterions.loc[criterion]["Value"]
                new_row[criterion] = value
            crit_values.loc[len(crit_values)] = new_row
    print(crit_values)
    crit_values.to_excel("crit_values_nonsorted.xlsx")
    crit_values.replace([np.inf, -np.inf], 99999999999, inplace=True)
    numeric_columns = crit_values.select_dtypes(include=[np.number]).columns

    crit_values[numeric_columns] = crit_values[numeric_columns].map(
        lambda x: "-" if pd.notnull(x) and x >= 1 else x)
    crit_values.to_excel("crit_values_sorted.xlsx")

    print(crit_values)

    crit_and_sens = pd.DataFrame(columns=["Criterion", "CritVal%", "SensVal"])
    for col in numeric_columns:
        numeric_rows = crit_values[col].apply(pd.to_numeric, errors='coerce')
        numeric_rows = numeric_rows.abs()

        new_row = {"Criterion": col, "CritVal%": numeric_rows.min() * 100, "SensVal": 1 / (numeric_rows.min() * 100)}
        crit_and_sens.loc[len(crit_and_sens)] = new_row
    crit_and_sens.to_excel("crit_and_sens.xlsx")
    warnings.resetwarnings()
    return crit_and_sens


if __name__ == '__main__':
    criterions_matrix = np.array([
        [    1,     3,          1, 1 / 3],
        [1 / 3,     1,      1 / 7, 1 / 3],
        [    1,     7,          1, 1 / 3],
        [    3,     3,          3,     1]
    ])

    flavanols_matrix = np.array([
        [    1,     3,     7,     9,     9],
        [1 / 3,     1,     7,     9,     9],
        [1 / 7, 1 / 7,     1,     5,     7],
        [1 / 9, 1 / 9, 1 / 5,     1,     3],
        [1 / 9, 1 / 9, 1 / 7, 1 / 3,     1]
    ])

    cost_matrix = np.array([
        [    1, 1 / 3,     3,     7,     3],
        [    3,     1,     3,     7,     3],
        [1 / 3, 1 / 3,     1,     7, 1 / 3],
        [1 / 7, 1 / 7, 1 / 7,     1, 1 / 7],
        [1 / 3, 1 / 3, 1 / 3,     7,     1]
    ])

    metals_matrix = np.array([
        [    1,     9,     5,     3,     5],
        [1 / 9,     1, 1 / 9, 1 / 9, 1 / 9],
        [1 / 5,     9,     1, 1 / 3, 1 / 3],
        [1 / 3,     9,     3,     1,     5],
        [1 / 5,     9,     3, 1 / 5,     1]
    ])

    delivery_matrix = np.array([
        [    1,     1,     5,     5, 1 / 5],
        [    1,     1,     5,     5, 1 / 5],
        [1 / 5, 1 / 5,     1,     1, 1 / 7],
        [1 / 5, 1 / 5,     1,     1, 1 / 7],
        [    5,     5,     7,     7,     1]
    ])
    #print(calculate_alternatives_weight(criterions_matrix,
    #                                     [flavanols_matrix,cost_matrix,metals_matrix,delivery_matrix],
    #                                     get_wq_norm_sum))

    #print(asses_consistency_of_hierarchy(criterions_matrix,
    #                                     [flavanols_matrix,cost_matrix,metals_matrix,delivery_matrix],
    #                                     get_wq_norm_geometric))

    print(get_crit_values(["Navitas","NOW","Ghirardelli","Valrhona","Hershey's"],
                                         ["Flavanols","Price","Metals","Delivery"],
                                         criterions_matrix,
                                         [flavanols_matrix, cost_matrix, metals_matrix, delivery_matrix],
                                         get_wq_norm_geometric))