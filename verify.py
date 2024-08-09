import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from DecisionTree import DecisionTree
from RandomForest import RandomForest

# Create a synthetic classification dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X = pd.DataFrame(X, columns=['feature1', 'feature2'])
y = np.array(y)

# Initialize the RandomForest class
rf = RandomForest(n_trees=3, max_depth=2, min_samples_split=2, treetype="classification", random_state=42)

# Fit the RandomForest with the dataset
rf.fit(X, y)

# New data point to add
new_data = pd.Series({'feature1': 0.5, 'feature2': 0.5})
new_label = 1

# Choose the first tree in the forest to update
tree_to_update = rf.trees[0]

# Extend the tree's y array to include the new label
tree_to_update.y = np.append(tree_to_update.y, new_label)

# Call traverse_add_path with the new data and its index
tree_to_update.traverse_add_path(new_data, x_index=len(X), y=new_label)

# Iterate over all nodes in node_id_dict and print the desired attributes
print("Tree structure after adding new data point:")
for node_id, node_info in tree_to_update.node_id_dict.items():
    node = node_info['node']  # Access the actual node object
    print(f"Node ID: {node_id}")
    print(f"  Sample Indices: {node.sample_indices}")
    print(f"  Value: {node.value}")
    print(f"  Gini: {node.gini}")
    
    # Check if the node has classification-specific attributes
    if hasattr(node, 'clf_value_dis'):
        print(f"  Value Distribution (clf_value_dis): {node.clf_value_dis}")
    if hasattr(node, 'clf_prob_dis'):
        print(f"  Probability Distribution (clf_prob_dis): {node.clf_prob_dis}")



'''
    Tree structure after adding new data point:
    Node ID: 0
    Sample Indices: [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
    18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
    36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
    54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
    72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
    90  91  92  93  94  95  96  97  98  99 100]
    Value: 1
    Gini: 0.4940692089010881
    Value Distribution (clf_value_dis): [45, 56]
    Probability Distribution (clf_prob_dis): [0.44554455 0.55445545]
    Node ID: 1
    Sample Indices: [ 1  2  9 10 11 17 20 22 23 26 28 29 31 32 33 39 40 42 43 46 47 48 49 50
    51 53 54 57 58 59 66 69 71 73 77 78 80 83 88 92 94 97 99]
    Value: 0
    Gini: 0.0
    Value Distribution (clf_value_dis): [43, 0]
    Probability Distribution (clf_prob_dis): [1. 0.]
    Node ID: 2
    Sample Indices: [  0   3   4   5   6   7   8  12  13  14  15  16  18  19  21  24  25  27
    30  34  35  36  37  38  41  44  45  52  55  56  60  61  62  63  64  65
    67  68  70  72  74  75  76  79  81  82  84  85  86  87  89  90  91  93
    95  96  98 100]
    Value: 1
    Gini: 0.06658739595719365
    Value Distribution (clf_value_dis): [2, 56]
    Probability Distribution (clf_prob_dis): [0.03448276 0.96551724]
    Node ID: 3
    Sample Indices: [ 4 12 25 44 86 98]
    Value: 1
    Gini: 0.4444444444444444
    Value Distribution (clf_value_dis): [2, 4]
    Probability Distribution (clf_prob_dis): [0.33333333 0.66666667]
    Node ID: 4
    Sample Indices: [  0   3   5   6   7   8  13  14  15  16  18  19  21  24  27  30  34  35
    36  37  38  41  45  52  55  56  60  61  62  63  64  65  67  68  70  72
    74  75  76  79  81  82  84  85  87  89  90  91  93  95  96 100]
    Value: 1
    Gini: 0.0
    Value Distribution (clf_value_dis): [0, 52]
    Probability Distribution (clf_prob_dis): [0. 1.]
'''