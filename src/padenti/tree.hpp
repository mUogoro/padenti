/******************************************************************************
 * Padenti Library
 *
 * Copyright (C) 2015  Daniele Pianu <daniele.pianu@ieiit.cnr.it>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 ******************************************************************************/

#ifndef __TREE_HPP
#define __TREE_HPP

#include <string>

/*!
 * \brief Class representing a Random Forests' Node
 *
 * \tparam FeatType type of feature entries and threshold
 * \tparam FeatDim dimension (i.e. number of entries) of the feature
 */
template <typename FeatType, unsigned int FeatDim>
class TreeNode
{
public:
  /*!
   * Index of the left child for the current node. The member is also used to define the
   * type of node (intermediate, leaf or uninitialized) and can take one of the following
   * values:
   * - -1 if the node is a leaf node (i.e. it has no children);
   * - -2 if the node is uninitialized;
   * - >0 if the node is an intermediate node, i.e. it has children.
   * The right children index node is given by the value m_leftChild+1.
   */
  int *m_leftChild;
  /*!
   * Values of the feature vector. For leaf nodes, the entries are set to zero.
   */
  FeatType *m_feature;
  /*!
   * Threshold used by the node splitting function (zero for leaf nodes).
   */
  FeatType *m_threshold;
  /*!
   * Posterior probability distribution over classed for the current node.
   */ 
  float *m_posterior;
};


/*!
 * \brief Class representing a single Random Forests binary tree.
 * The tree class stores the data associated to a single random forests binary tree. Internally,
 * the nodes left index, feature vector, threshold and posterior probability are stored 
 * continuously on separated arrays. The data of a singular node can be aggragated into a
 * TreeNode instance and returned to the user. Trees are associated with a unique ID and can be
 * loaded from/stored to disk.
 *
 * \tparam FeatType type of feature entries and threshold
 * \tparam FeatDim dimension (i.e. number of entries) of the feature
 * \tparam nClasses number of classes
 */
template <typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
class Tree
{
private:
  unsigned int m_id;
  unsigned int m_depth;
  TreeNode<FeatType, FeatDim> *m_nodes;

  int *m_leftChildren;
  FeatType *m_features;
  FeatType *m_thresholds;
  float *m_posteriors;

  void _init();
  void _clean();
public:
  /*!
   * Default constructor. An empty tree is created.
   */
  Tree();
  /*!
   * Create a new tree with id id and depth depth. Nodes are left uninitialized. The total number
   * of nodes is given by 2^(depth-1)-1.
   * 
   * \param id tree unique id
   * \param depth tree depth.
   */
  Tree(unsigned int id, unsigned int depth);
  ~Tree();

  /*!
   * Get the tree unique ID.
   *
   * \return the tree ID
   */
  unsigned int getID() const;

  /*!
   * Get the tree depth.
   *
   * \return the tree depth
   */
  unsigned int getDepth() const;

  /*!
   * Get the node with index idx.
   *
   * \param idx unique of the node to retrieve
   * \return a new TreeNode instance which stores the data of idx-th node
   */
  const TreeNode<FeatType, FeatDim>& getNode(unsigned int idx) const;

  /*!
   * Get the internal pointer to the vector of nodes left child index.
   *
   * \return the pointer to the nodes left child vector
   *
   */
  int *getLeftChildren() const;

  /*!
   * Get the internal pointer to the vector of nodes feature vector.
   *
   * \return the pointer to the nodes feature vector
   *
   */
  FeatType *getFeatures() const;

  /*!
   * Get the internal pointer to the vector of nodes threshold
   *
   * \return the pointer to the nodes threshold
   *
   */
  FeatType *getThresholds() const;

  /*!
   * Get the internal pointer to the vector of nodes posteriors distribution
   *
   * \return the pointer to the nodes posteriors distribution
   *
   */
  float *getPosteriors() const;

  /*!
   * Load a previously saved tree from disk. If more than one tree is stored in the file,
   * a specific tree can be selected by its id using the idx parameter
   *
   * \param treePath the path of the tree file
   * \param idx optional id of the tree to load
   */
  void load(const std::string &treePath, int idx=-1);

  /*!
   * Save the current tree to disk
   *
   * \param treePath the path of the tree file
   * \param idx optional id, different from the current one, used when saving the tree
   */
  void save(const std::string &treePath, int idx=-1) const;
};


#include <padenti/tree_impl.hpp>

#endif // __TREE_HPP
