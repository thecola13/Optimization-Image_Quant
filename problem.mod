/* Parameters */
param n;     /* Number of data points */
param P;     /* Number of possible clusters */
param k;     /* Number of clusters to select */

/* Cost for each cluster */
param c{1..P}; /* Cost associated with each cluster */

/* Cluster membership: C[j, i] = 1 if point i is in cluster j */
param C{1..P, 1..n}, binary;

/* Decision Variables */
var y{1..P}, binary; /* y[j] = 1 if cluster j is selected, 0 otherwise */

/* Objective Function */
minimize total_cost: sum{j in 1..P} c[j] * y[j];

/* Constraints */
/* Ensure each point is assigned to exactly one selected cluster */
s.t. assign_points{i in 1..n}:
    sum{j in 1..P} y[j] * C[j, i] = 1;

/* Ensure exactly k clusters are chosen */
s.t. select_k_clusters:
    sum{j in 1..P} y[j] = k;

solve;

/* Output */
printf "Total cost: %f\n", total_cost;
for {j in 1..P: y[j] = 1} {
    for {i in 1..n} {
        printf "%d ", C[j, i];
    }
    printf " (Cluster %d) ", j;
    printf "\n";
}
end;

