from pomegranate import *

# Section C
A = DiscreteDistribution({'under 18': 0.5, '18-59': 0.3, '+60': 0.2})
FI = ConditionalProbabilityTable(
        [['under 18', 'Promising', 0.9],
         ['under 18', 'Not promising', 0.1],
         ['18-59', 'Promising', 0.5],
         ['18-59', 'Not promising', 0.5],
         ['+60', 'Promising', 0.1],
         ['+60', 'Not promising', 0.9]], [A])
HD = DiscreteDistribution({'Low': 0.5, 'High': 0.5})
CW = ConditionalProbabilityTable(
        [['Promising', 'Low', 'Positive', 1.0],
         ['Promising', 'Low', 'Negative', 0.0],
         ['Promising', 'High', 'Positive', 0.5],
         ['Promising', 'High', 'Negative', 0.5],
         ['Not promising', 'Low', 'Positive', 0.5],
         ['Not promising', 'Low', 'Negative', 0.5],
         ['Not promising', 'High', 'Positive', 0.0],
         ['Not promising', 'High', 'Negative', 1.0]], [FI, HD])

s1 = Node(A, name="Age")
s2 = Node(FI, name="Future Income")
s3 = Node(HD, name="Historical Debts")
s4 = Node(CW, name="Creditworthiness")

model = BayesianNetwork("Credit Score")
model.add_states(s1, s2, s3, s4)
model.add_edge(s1, s2)
model.add_edge(s2, s4)
model.add_edge(s3, s4)
model.bake()


# Section D
print('-------- Section D ---------')
p_promising_given_60 = model.probability([['+60', 'Promising', None, None]]) / model.probability([['+60', None, None, None]])  # By Conditional probability formula
print('P(FI = P romosing|Age = +60)=', p_promising_given_60)


# Section E
print('-------- Section E ---------')
print('P(HD = Low|CW = Positive, FI = Promising)=', model.probability([[None, 'Promising', 'Low', 'Positive']]))
print('P(HD = High|CW = Positive, FI = Promising)=', model.probability([[None, 'Promising', 'High', 'Positive']]))
print('P(HD = Low|CW = Positive)=', model.probability([[None, None, 'Low', 'Positive']]))
print('P(HD = High|CW = Positive)=', model.probability([[None, None, 'High', 'Positive']]))


# Section F
print('-------- Section F ---------')
A = DiscreteDistribution({'under 18': 0.5, '18-59': 0.3, '+60': 0.2})
FI = ConditionalProbabilityTable(
        [['under 18', 'Promising', 0.9],
         ['under 18', 'Not promising', 0.1],
         ['18-59', 'Promising', 0.5],
         ['18-59', 'Not promising', 0.5],
         ['+60', 'Promising', 0.1],
         ['+60', 'Not promising', 0.9]], [A])
HD = DiscreteDistribution({'Low': 0.5, 'High': 0.5})
CW = ConditionalProbabilityTable(
        [['Promising', 'Low', 'Positive', 1.0],
         ['Promising', 'Low', 'Negative', 0.0],
         ['Promising', 'High', 'Positive', 1.0],
         ['Promising', 'High', 'Negative', 0.0],
         ['Not promising', 'Low', 'Positive', 0.0],
         ['Not promising', 'Low', 'Negative', 1.0],
         ['Not promising', 'High', 'Positive', 0.0],
         ['Not promising', 'High', 'Negative', 1.0]], [FI, HD])

s1 = Node(A, name="Age")
s2 = Node(FI, name="Future Income")
s3 = Node(HD, name="Historical Debts")
s4 = Node(CW, name="Creditworthiness")

model = BayesianNetwork("Credit Score")
model.add_states(s1, s2, s3, s4)
model.add_edge(s1, s2)
model.add_edge(s2, s4)
model.add_edge(s3, s4)
model.bake()

print('P(HD = Low|CW = Positive, FI = Promising)=', model.probability([[None, 'Promising', 'Low', 'Positive']]))
print('P(HD = High|CW = Positive, FI = Promising)=', model.probability([[None, 'Promising', 'High', 'Positive']]))
print('P(HD = Low|CW = Positive)=', model.probability([[None, None, 'Low', 'Positive']]))
print('P(HD = High|CW = Positive)=', model.probability([[None, None, 'High', 'Positive']]))
