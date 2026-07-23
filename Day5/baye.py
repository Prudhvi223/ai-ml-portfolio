# Prior probability
P_disease = 0.01

# Probability of positive test if disease exists
P_positive_given_disease = 0.95

# Overall probability of positive test
P_positive = 0.02

# Bayes theorem
P_disease_given_positive = (
    P_positive_given_disease * P_disease
) / P_positive

print("Probability of disease after a positive test:")
print(P_disease_given_positive)