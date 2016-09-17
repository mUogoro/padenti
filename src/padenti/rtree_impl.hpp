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

#include <algorithm>
#include <string>
#include <sstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <padenti/rtree.hpp>


template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
RTree<FeatType, FeatDim, RDim>::RTree():
  m_id(0), m_depth(0),
  m_nodes(NULL), m_leftChildren(NULL),
  m_features(NULL), m_thresholds(NULL), m_values(NULL)
{}

template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
RTree<FeatType, FeatDim, RDim>::RTree(unsigned int id, unsigned int depth):
  m_id(id), m_depth(depth)
{
  _init();
}

template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
RTree<FeatType, FeatDim, RDim>::~RTree()
{
  _clean();
}

template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
unsigned int RTree<FeatType, FeatDim, RDim>::getID() const
{
  return m_id;
}

template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
unsigned int RTree<FeatType, FeatDim, RDim>::getDepth() const
{
  return m_depth;
}

template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
const RTreeNode<FeatType, FeatDim, RDim>& RTree<FeatType, FeatDim, RDim>::getNode(unsigned int idx) const
{
  return m_nodes[idx];
}

template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
int *RTree<FeatType, FeatDim, RDim>::getLeftChildren() const
{
  return m_leftChildren;
}

template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
FeatType *RTree<FeatType, FeatDim, RDim>::getFeatures() const
{
  return m_features;
}

template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
FeatType *RTree<FeatType, FeatDim, RDim>::getThresholds() const
{
  return m_thresholds;
}

template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
float *RTree<FeatType, FeatDim, RDim>::getValues() const
{
  return m_values;
}


template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
void RTree<FeatType, FeatDim, RDim>::load(const std::string &treePath, int idx)
{  
  boost::property_tree::ptree pt;
  boost::property_tree::read_xml(treePath, pt);
  
  if (idx!=-1)
  {
    m_id = idx;
  }
  unsigned int currDepth = pt.get<unsigned int>("Trees.MaxDepth");

  if (!m_depth || m_depth!=currDepth)
  {
    _clean();

    m_depth = currDepth;
    _init();
  }

  for (unsigned int i=0; i<((2<<(m_depth-1))-1); i++)
  {
    std::stringstream nodeStream;
    const RTreeNode<FeatType, FeatDim, RDim> &currNode = getNode(i);
    
    nodeStream << "Trees.Tree" << m_id << ".Node" << i << ".LeftChild";
    *currNode.m_leftChild = pt.get<int>(nodeStream.str(), -2);

    if (*currNode.m_leftChild==-2)
    {
      // Node does not exists: fill with zeros
      std::fill_n(currNode.m_feature, FeatDim, 0);
      *currNode.m_threshold = 0;
      std::fill_n(currNode.m_value, RDim, 0);
    }
    else if (*currNode.m_leftChild==-1)
    {
      // Leaf node
      std::stringstream valueStream;

      nodeStream.str("");
      nodeStream << "Trees.Tree" << m_id << ".Node" << i << ".Value";      
      valueStream.str(pt.get<std::string>(nodeStream.str()));
      for (int l=0; l<RDim; l++)
      {
	valueStream >> currNode.m_value[l];
      }

      std::fill_n(currNode.m_feature, FeatDim, 0);
      *currNode.m_threshold = 0;
    }
    else
    {
      // Split Node
      std::stringstream valueStream;
      std::stringstream featuresStream;

      nodeStream.str("");
      nodeStream << "Trees.Tree" << m_id << ".Node" << i << ".Value";
      valueStream.str(pt.get<std::string>(nodeStream.str()));
      for (int l=0; l<RDim; l++)
      {
	valueStream >> currNode.m_value[l];
      }

      nodeStream.str("");
      nodeStream << "Trees.Tree" << m_id << ".Node" << i << ".SplitParameters";
      featuresStream.str(pt.get<std::string>(nodeStream.str()));
      for (int j=0; j<FeatDim; j++)
      {
	featuresStream >> currNode.m_feature[j];
      }
      featuresStream >> *currNode.m_threshold;
    }
  }

  // Done
}

/**
 * \todo new tree's xml format (e.g. add training information)
 * \todo tree zipping??? (i.e. shift nodes when inner nodes with -2 leftChild member are found)
 */
template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
void RTree<FeatType, FeatDim, RDim>::save(const std::string &treePath, int idx) const
{
  boost::property_tree::ptree pt;
  std::stringstream nodeStream;

  pt.put("Trees.TreeNumber", 1);
  pt.put("Trees.MaxDepth", m_depth);

  for (unsigned int i=0; i<((2<<(m_depth-1))-1); i++)
  {
    const RTreeNode<FeatType, FeatDim, RDim> &currNode = getNode(i);

    if (*currNode.m_leftChild==-2) continue;

    nodeStream.clear();
    nodeStream.str("");
    nodeStream << "Trees.Tree" << ((idx!=-1) ? idx : m_id) << ".Node" << i;
    pt.add(nodeStream.str()+".LeftChild", *currNode.m_leftChild);
    
    if (*currNode.m_leftChild != -1)
    {
      // Split node
      std::stringstream featThrStream;

      for (unsigned int f=0; f<FeatDim; f++)
      {
	featThrStream << currNode.m_feature[f]  << " ";
      }
      featThrStream << (*currNode.m_threshold);

      pt.add(nodeStream.str()+".SplitParameters", featThrStream.str());
    }

    std::stringstream valueStream;

    for (unsigned int l=0; l<RDim-1; l++)
    {
      valueStream << currNode.m_value[l] << " ";
    }
    valueStream << currNode.m_value[RDim-1];

    pt.add(nodeStream.str()+".Value", valueStream.str());
  }

  boost::property_tree::write_xml(treePath, pt);
}


template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
void RTree<FeatType, FeatDim, RDim>::_init()
{
  unsigned int nNodes = (2<<(m_depth-1))-1;
  
  // Allocate tree data. Tree nodes members are allocated separately on continuos
  // vectors. This allow to easily pass tree nodes' data to GPU without dealing
  // with host-GPU memory alignment problems when using array-of-structs to represent
  // the tree
  m_leftChildren = new int[nNodes];
  m_features = new FeatType[nNodes*FeatDim];
  m_thresholds = new FeatType[nNodes];
  m_values = new float[nNodes*RDim];

  // To simplify node data access, provide an array-of-struct representation of the tree:
  // structs members are pointers to the continuosly allocated data
  m_nodes = new RTreeNode<FeatType, FeatDim, RDim>[nNodes];
  for (int i=0; i<nNodes; i++)
  {
    RTreeNode<FeatType, FeatDim, RDim> *currNode = &m_nodes[i];

    currNode->m_leftChild = &m_leftChildren[i];
    currNode->m_feature = &m_features[i*FeatDim];
    currNode->m_threshold = &m_thresholds[i];
    currNode->m_value = &m_values[i*RDim];

    // Init tree nodes:
    // - -2 left child
    // - zeroed feature, threshold and posteriors
    *currNode->m_leftChild = -2;
    std::fill_n(currNode->m_feature, FeatDim, (FeatType)0);
    *currNode->m_threshold = (FeatType)0;
    std::fill_n(currNode->m_value, RDim, 0.0f);
  }
}

template <typename FeatType, unsigned int FeatDim, unsigned int RDim>
void RTree<FeatType, FeatDim, RDim>::_clean()
{
  delete []m_nodes;
  delete []m_values;
  delete []m_thresholds;
  delete []m_features;
  delete []m_leftChildren;
}
