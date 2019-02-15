#!/usr/bin/python
"""
Octree implementation
"""
# From: https://code.google.com/p/pynastran/source/browse/trunk/pyNastran/general/octree.py?r=949
#       http://code.activestate.com/recipes/498121-python-octree-implementation/

# UPDATED:
# Is now more like a true octree (ie: partitions space containing objects)

# Important Points to remember:
# The OctNode positions do not correspond to any object position
# rather they are seperate containers which may contain objects
# or other nodes.

# An OctNode which which holds less objects than MAX_OBJECTS_PER_CUBE
# is a LeafNode; it has no branches, but holds a list of objects contained within
# its boundaries. The list of objects is held in the leafNode's 'data' property

# If more objects are added to an OctNode, taking the object count over MAX_OBJECTS_PER_CUBE
# Then the cube has to subdivide itself, and arrange its objects in the new child nodes.
# The new octNode itself contains no objects, but its children should.

from __future__ import print_function
#import matplotlib.pyplot as plt

try:
    import numpy as np
except ImportError:
    np = None

#dic={}
totnum=0

class OctNode(object):
    """
    New Octnode Class, can be appended to as well i think
    """
    def __init__(self, position, size, depth, data):
        """
        OctNode Cubes have a position and size
        position is related to, but not the same as the objects the node contains.

        Branches (or children) follow a predictable pattern to make accesses simple.
        Here, - means less than 'origin' in that dimension, + means greater than.
        branch: 0 1 2 3 4 5 6 7
        x:      - - - - + + + +
        y:      - - + + - - + +
        z:      - + - + - + - +
        """
        self.position = position
        self.size = size
        self.depth = depth

        ## All OctNodes will be leaf nodes at first
        ## Then subdivided later as more objects get added
        self.isLeafNode = True

        ## store our object, typically this will be one, but maybe more
        self.data = data

        ## might as well give it some emtpy branches while we are here.
        self.branches = [None, None, None, None, None, None, None, None]

        half = size / 2

        ## The cube's bounding coordinates
        self.lower = (position[0] - half, position[1] - half, position[2] - half)
        self.upper = (position[0] + half, position[1] + half, position[2] + half)

    def __str__(self):
        data_str = u", ".join((str(x) for x in self.data))
        return u"position: {0}, size: {1}, depth: {2} leaf: {3}, data: {4}".format(
            self.position, self.size, self.depth, self.isLeafNode, data_str
        )


class Octree(object):
    """
    The octree itself, which is capable of adding and searching for nodes.
    """
    def __init__(self, worldSize, origin=(0, 0, 0), max_type="nodes", max_value=10):
        """
        Init the world bounding root cube
        all world geometry is inside this
        it will first be created as a leaf node (ie, without branches)
        this is because it has no objects, which is less than MAX_OBJECTS_PER_CUBE
        if we insert more objects into it than MAX_OBJECTS_PER_CUBE, then it will subdivide itself.

        """
        self.root = OctNode(origin, worldSize, 0, [])
        self.worldSize = worldSize
        self.limit_nodes = (max_type=="nodes")
        self.limit = max_value

    @staticmethod
    def CreateNode(position, size, objects):
        """This creates the actual OctNode itself."""
        return OctNode(position, size, objects)

    def insertNode(self, position, objData=None):
        """
        Add the given object to the octree if possible

        Parameters
        ----------
        position : array_like with 3 elements
            The spatial location for the object
        objData : optional
            The data to store at this position. By default stores the position.

            If the object does not have a position attribute, the object
            itself is assumed to be the position.

        Returns
        -------
        node : OctNode or None
            The node in which the data is stored or None if outside the
            octree's boundary volume.

        """
        if np:
            if np.any(position < self.root.lower):
                return None
            if np.any(position > self.root.upper):
                return None
        else:
            if position < self.root.lower:
                return None
            if position > self.root.upper:
                return None

        if objData is None:
            objData = position

        return self.__insertNode(self.root, self.root.size, self.root, position, objData)

    def __insertNode(self, root, size, parent, position, objData):
        """Private version of insertNode() that is called recursively"""
        if (root is None) :
            # we're inserting a single object, so if we reach an empty node, insert it here
            # Our new node will be a leaf with one object, our object
            # More may be added later, or the node maybe subdivided if too many are added
            # Find the Real Geometric centre point of our new node:
            # Found from the position of the parent node supplied in the arguments
            pos = parent.position

            ## offset is halfway across the size allocated for this node
            offset = size / 2

            ## find out which direction we're heading in
            branch = self.__findBranch(parent, position)

            ## new center = parent position + (branch direction * offset)
            newCenter = (0, 0, 0)

            if branch == 0:
                newCenter = (pos[0] - offset, pos[1] - offset, pos[2] - offset )
            elif branch == 1:
                newCenter = (pos[0] - offset, pos[1] - offset, pos[2] + offset )
            elif branch == 2:
                newCenter = (pos[0] - offset, pos[1] + offset, pos[2] - offset )
            elif branch == 3:
                newCenter = (pos[0] - offset, pos[1] + offset, pos[2] + offset )
            elif branch == 4:
                newCenter = (pos[0] + offset, pos[1] - offset, pos[2] - offset )
            elif branch == 5:
                newCenter = (pos[0] + offset, pos[1] - offset, pos[2] + offset )
            elif branch == 6:
                newCenter = (pos[0] + offset, pos[1] + offset, pos[2] - offset )
            elif branch == 7:
                newCenter = (pos[0] + offset, pos[1] + offset, pos[2] + offset )

            # Now we know the centre point of the new node
            # we already know the size as supplied by the parent node
            # So create a new node at this position in the tree
            #print("Adding Node of size: " + str(size / 2) + " at " + str(newCenter))
            return OctNode(newCenter, size, parent.depth + 1, [objData])

        #else: are we not at our position, but not at a leaf node either
        elif (
            not root.isLeafNode
            #and
            #(
            #    (np and np.any(root.position != position))
            #    or
            #    (root.position != position)
            #)
        ):
            # we're in an octNode still, we need to traverse further
            branch = self.__findBranch(root, position)
            # Find the new scale we working with
            newSize = root.size / 2
            # Perform the same operation on the appropriate branch recursively
            #print(root.branches[branch])
                
            root.branches[branch] = self.__insertNode(root.branches[branch], newSize, root, position, objData)

        # else, is this node a leaf node with objects already in it?
        elif root.isLeafNode:
        #else:
            # We've reached a leaf node. This has no branches yet, but does hold
            # some objects, at the moment, this has to be less objects than MAX_OBJECTS_PER_CUBE
            # otherwise this would not be a leafNode (elementary my dear watson).
            # if we add the node to this branch will we be over the limit?
            if (
                (self.limit_nodes and len(root.data) < self.limit)
                or
                (not self.limit_nodes and root.depth >= self.limit)
            ):
                # No? then Add to the Node's list of objects and we're done
                #print(root)
                root.data.append(objData)
                #return root
            else:
                # Adding this object to this leaf takes us over the limit
                # So we have to subdivide the leaf and redistribute the objects
                # on the new children.
                # Add the new object to pre-existing list
                root.data.append(objData)
                # copy the list
                objList = root.data
                # Clear this node's data
                root.data = None
                # It is not a leaf node anymore
                root.isLeafNode = False
                # Calculate the size of the new children
                newSize = root.size / 2
                # distribute the objects on the new tree
                #print ("Subdividing Node sized at: " + str(root.size) + " at " + str(root.position))
                for indd, ob in enumerate(objList):
                    # Use the position attribute of the object if possible
                    if hasattr(ob, "position"):
                        pos = ob.position
                    else:
                        pos = ob
                    branch = self.__findBranch(root, pos)
                    root.branches[branch] = self.__insertNode(root.branches[branch], newSize, root, pos, ob)
        return root

    def findPosition(self, position):
        """
        Basic lookup that finds the leaf node containing the specified position
        Returns the child objects of the leaf, or None if the leaf is empty or none
        """
        if np:
            if np.any(position < self.root.lower):
                return None
            if np.any(position > self.root.upper):
                return None
        else:
            if position < self.root.lower:
                return None
            if position > self.root.upper:
                return None
        return self.__findPosition(self.root, position)

    @staticmethod
    def __findPosition(node, position, count=0, branch=0):
        """Private version of findPosition """
        if node.isLeafNode:
            #print("The position is", position, " data is", node.data)
            return node.data
        branch = Octree.__findBranch(node, position)
        child = node.branches[branch]
        if child is None:
            return None
        return Octree.__findPosition(child, position, count + 1, branch)

    @staticmethod
    def __findBranch(root, position):
        """
        helper function
        returns an index corresponding to a branch
        pointing in the direction we want to go
        """
        index = 0
        if (position[0] >= root.position[0]):
            index |= 4
        if (position[1] >= root.position[1]):
            index |= 2
        if (position[2] >= root.position[2]):
            index |= 1
        return index

    def iterateDepthFirst(self):
        """Iterate through the octree depth-first"""
        gen = self.__iterateDepthFirst(self.root)
        for n in gen:
            yield n

    @staticmethod
    def __iterateDepthFirst(root):
        """Private (static) version of iterateDepthFirst"""

        for branch in root.branches:
            if branch is None:
                continue
            for n in Octree.__iterateDepthFirst(branch):
                yield n
            if branch.isLeafNode:
                yield branch
    

    def myIterate(self, dic, cells):
        pp=[0]
        self._myIterate(dic, self.root, pp,cells)

    @staticmethod
    def _myIterate(dic, root, path, cells):
        for i, branch in enumerate(root.branches):
            if branch is None:
                continue
            #for i, n in Octree._myIterate(branch, depth+1, path.append(i)): 
            #    yield n
            pn=path[:]
            pn.append(i)
            Octree._myIterate(dic, branch, pn, cells)
            if branch.isLeafNode:
                #cells[int(branch.objData.name)]=path
                #print(branch)
                index=0
                global totnum
                for i in branch.data:
                    totnum=totnum+1
                    index = 0
                    for ind, j in enumerate(path):
                        index = (index<<3)+j
                        if (ind,index) in dic:
                            dic[(ind, index)].append(i.name)
                        else:
                            dic[(ind, index)]=[i.name]


class TestObject(object):
        """Dummy object class to test with"""
        def __init__(self, name, position):
            self.name = name
            self.position = position

        def __str__(self):
            return u"name: {0} position: {1}".format(self.name, self.position)

## ---------------------------------------------------------------------------------------------------##
#indexArray: Nx1 arrays of tuples (x, y, z) represents point index. e.g., indexArray[0]=(0, 2.3, 3.4)
#max_node: the maximum number of points in the leaf node of octree
#max_depth: maximum depth of octree
#max_size: the octree range
#center: center position, e.g., (0,0,0)
def octree_func( dic, indexArray, max_node, max_depth, max_size, center ):
        # Create a new octree, size of world
        N = len(indexArray)
        #print(indexArray) 
        myTree = Octree(
            max_size,    #maximum index range
            center,
            'depth',
            max_depth
        )

        # Insert some random objects and time it
        obj = []
        for index, (x,y,z) in enumerate(indexArray):
            #print ( indexArray[index])
            myObject = TestObject(str(index), indexArray[index])
            myTree.insertNode(myObject.position, myObject)

        #for i, x in enumerate(myTree.iterateDepthFirst()):
            #print(i, ":", x)
        cells=[]*N
        myTree.myIterate(dic, cells)


# def octree_func( dic, indexArray, max_node, max_depth, max_size, center ):
#         # Create a new octree, size of world
#         N = len(indexArray)
#         #print(indexArray) 
#         myTree = Octree(
#             max_size,    #maximum index range
#             center,
#             'nodes',
#             max_node
#         )

#         # Insert some random objects and time it
#         obj = []
#         for index, (x,y,z) in enumerate(indexArray):
#             #print ( indexArray[index])
#             myObject = TestObject(str(index), indexArray[index])
#             myTree.insertNode(myObject.position, myObject)

#         #for i, x in enumerate(myTree.iterateDepthFirst()):
#             #print(i, ":", x)
#         cells=[]*N
#         myTree.myIterate(dic, cells)


# testing code
### Object Insertion Test ###

# # So lets test the adding:
# import random
# import time
# # Number of objects we intend to add.
# NUM_TEST_OBJECTS = 2000

# # Number of lookups we're going to test
# NUM_LOOKUPS = 2000

# # Size that the octree covers
# WORLD_SIZE = 10000.0

# ORIGIN = (WORLD_SIZE/2, WORLD_SIZE/2, WORLD_SIZE/2)
# #ORIGIN = (0, 0, 0)

# # The range from which to draw random values
# #RAND_RANGE = (-WORLD_SIZE * 0.3, WORLD_SIZE * 0.3)
# RAND_RANGE =(0, WORLD_SIZE)


# # create random test objects
# Points =np.random.rand(NUM_TEST_OBJECTS,3)*WORLD_SIZE
# Points_set = tuple(map(tuple, Points))


# octree_dict = {}
# octree_func(octree_dict, Points_set, NUM_TEST_OBJECTS, 2, WORLD_SIZE, ORIGIN)

# all_features = []
# for key, value in octree_dict.items():
#     point_subspace = Points[np.asarray(value, dtype=np.int32),:]
#     # feature_subspace = Pointnet(point_subspace)
#     # all_features.append(feature_subspace)
#     print(key, point_subspace.shape)
