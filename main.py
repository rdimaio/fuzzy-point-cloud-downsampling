import numpy as np
from scipy import stats

IMPORTANCE_THRESHOLD = 0.8

# input_cloud is a numpy array of shape [n_points, n_dims]

# Perform density estimation
kde = stats.gaussian_kde(input_cloud.T)

# Evaluate KDE on input cloud
input_density = kde(input_cloud.T)

# x_density is the density value of a point
# x_distance is the distance value of a point

class TriangularMembershipFunction:
    """ Fuzzy number represented using a triangular membership function.
    
    Parameters:
        a: x-coordinate of leftmost vertex of triangle
        b: x-coordinate of upper vertex of triangle (i.e. where membership = 1)
        c: x-coordinate of rightmost vertex of triangle
    """

    def __init__(self, a, b, c):
        assert a <= b and b <= c, 'a, b and c must not be equal and must be in increasing order.'
        self.a = a
        self.b = b
        self.c = c
        self.slope = 1/(b-a)

    def fuzzify(self, x):
        """Fuzzify an input. Returns the degree of truth for that input.
        
        Parameters:
            x: crisp input
        """
        # Membership is 0 if x is not within the range of the membership function
        if x < self.a or x > self.c:
            return 0
        # If x is on the left side of the membership function
        if x < self.b:
            return self.y_left(x)
        else:
            return self.y_right(x)
        
    def y_left(self, x):
        # y - y1 = m(x - x1)
        #   x1, y1 are the coordinates of b
        y = self.slope * (x - self.b) + 1
        return y

    def y_right(self, x):
        # y - y1 = -m(x - x1)
        #   x1, y1 are the coordinates of b
        #   same as y_left, but slope is negative
        y = -self.slope * (x - self.b) + 1
        return y
    
    def alpha_cut(self, y):
        """Given a membership value, returns x coordinates at which that membership is achieved.
        e.g. if a=0,b=0.25,c=0.5 and y=0.9, it returns 0.225 and 0.275.
        at x=0.225 and x=0.275, the membership value is 0.9.
        
        Parameters:
            y: membership value (i.e. degree of truth)"""

        assert y >= 0 and y <= 1, 'degree of truth must be between 0 and 1.'

        x_left = self.b - ((y - 1) / self.slope)

        x_right = self.b + ((y - 1) / self.slope)

        return x_left, x_right


# Density membership functions
empty = TriangularMembershipFunction(-0.25, 0.00, 0.25)
sparse = TriangularMembershipFunction(0.00, 0.25, 0.50)
uniform = TriangularMembershipFunction(0.25, 0.50, 0.75)
dense = TriangularMembershipFunction(0.50, 0.75, 1.00)
full = TriangularMembershipFunction(0.75, 1.00, 1.25)

# Distance membership functions
# Effectively the same as the density membership functions, but re-declared to maintain consistency
very_close = TriangularMembershipFunction(-0.25, 0.00, 0.25)
close = TriangularMembershipFunction(0.00, 0.25, 0.50)
halfway = TriangularMembershipFunction(0.25, 0.50, 0.75)
far = TriangularMembershipFunction(0.50, 0.75, 1.00)
very_far = TriangularMembershipFunction(0.75, 1.00, 1.25)

# Output importance membership functions
superfluous = TriangularMembershipFunction(0.00, 0.25, 0.50)
important = TriangularMembershipFunction(0.25, 0.50, 0.75)
essential = TriangularMembershipFunction(0.50, 0.75, 1.00)

superfluous_firing_strength = 0.00
important_firing_strength = 0.00
essential_firing_strength = 0.00


# Rules
if empty.fuzzify(x_density) > 0 and (very_close.fuzzify(x_distance) > 0 or close.fuzzify(x_distance) > 0 or halfway.fuzzify(x_distance) > 0):
    
    # Take the maximum membership out of the distance conditions as they are separated by OR
    distance_membership = max(very_close.fuzzify(x_distance), close.fuzzify(x_distance), halfway.fuzzify(x_distance))

    # Take the minimum membership between density and distance condition as firing strength
    firing_strength = min(empty.fuzzify(x_density), distance_membership)

    essential_firing_strength = firing_strength
    print("x is essential")

elif empty.fuzzify(x_density) > 0 and (far.fuzzify(x_distance) or very_far.fuzzify(x_distance)):
    # Take the maximum membership out of the distance conditions as they are separated by OR
    distance_membership = max(far.fuzzify(x_distance), very_far.fuzzify(x_distance))

    # Take the minimum membership between density and distance condition as firing strength
    firing_strength = min(empty.fuzzify(x_density), distance_membership)

    superfluous_firing_strength = firing_strength
    print("x is superfluous")

if sparse.fuzzify(x_density) > 0 or uniform.fuzzify(x_density) > 0:

    density_membership = max(sparse.fuzzify(x_density), uniform.fuzzify(x_density))

    if very_close.fuzzify(x_distance) > 0 or close.fuzzify(x_distance) > 0:

        distance_membership = max(very_close.fuzzify(x_distance), close.fuzzify(x_distance))

        firing_strength = min(density_membership, distance_membership)

        # Because we are doing a union of the different rules,
        # We want to take the max. firing strength for each given output membership function
        #   (i.e. the maximum firing strength for superfluous, important, essential)
        # So if a bigger firing strength is achieved here, update the firing strength for essential.
        if (firing_strength > essential_firing_strength):
            essential_firing_strength = firing_strength
        print("x is essential")

    if halfway.fuzzify(x_distance) > 0:
        firing_strength = min(density_membership, halfway.fuzzify(x_distance))

        if (firing_strength > important_firing_strength):
            important_firing_strength = firing_strength
        print("x is important")

    if far.fuzzify(x_distance) > 0 or very_far.fuzzify(x_distance) > 0:

        distance_membership = max(far.fuzzify(x_distance), very_far.fuzzify(x_distance))

        firing_strength = min(density_membership, distance_membership)

        if (firing_strength > superfluous_firing_strength):
            superfluous_firing_strength = firing_strength
        print("x is superfluous")

if dense.fuzzify(x_density):

    density_membership = dense.fuzzify(x_density)

    if very_close.fuzzify(x_distance) or close.fuzzify(x_distance):
        distance_membership = max(very_close.fuzzify(x_distance), close.fuzzify(x_distance))

        firing_strength = min(density_membership, distance_membership)

        if (firing_strength > important_firing_strength):
            important_firing_strength = firing_strength
        print("x is important")

    else:
        firing_strength = density_membership
        if (firing_strength > superfluous_firing_strength):
            superfluous_firing_strength = firing_strength
        print("x is superfluous")

if full.fuzzify(x_density) or far.fuzzify(x_distance) or very_far.fuzzify(x_distance):
    firing_strength = max(full.fuzzify(x_density), far.fuzzify(x_distance), very_far.fuzzify(x_distance))
    if (firing_strength > superfluous_firing_strength):
        superfluous_firing_strength = firing_strength
    print("x is superfluous")

# Defuzzification
# First, let's do a union of all the active output membership functions.
# To do so, we sample 8 points from 0 to 1 at a step size of 0.15.
points = np.zeros([8, 2]) # 8 points, 2 dimensions each
# (2 dimensions because we are dealing with the dimensions of the membership functions,
# not with the points in the input point cloud)

# Only need to find max firing strength if an x coordinate corresponds to multiple possible membership values > 0
points[0] = (0.00, 0.00)
points[1] = (0.15, superfluous_firing_strength)
points[2] = (0.30, max(superfluous_firing_strength, important_firing_strength))
points[3] = (0.45, max(superfluous_firing_strength, important_firing_strength))
points[4] = (0.60, max(important_firing_strength, essential_firing_strength))
points[5] = (0.60, max(important_firing_strength, essential_firing_strength))
points[6] = (0.75, essential_firing_strength)
points[7] = (1.00, 0.00)

# Calculate center of gravity (CoA)
defuzzified_importance = (np.dot(points[:, 0], points[:, 1])) / np.sum(points[:, 1])

downsampled_point_cloud = input_cloud[np.where(defuzzified_importance > IMPORTANCE_THRESHOLD)]