//
// Copyright 2014 Mitsubishi Electric Research Laboratories All
// Rights Reserved.
//
// Permission to use, copy and modify this software and its
// documentation without fee for educational, research and non-profit
// purposes, is hereby granted, provided that the above copyright
// notice, this paragraph, and the following three paragraphs appear
// in all copies.
//
// To request permission to incorporate this software into commercial
// products contact: Director; Mitsubishi Electric Research
// Laboratories (MERL); 201 Broadway; Cambridge, MA 02139.
//
// IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT,
// INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
// LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
// DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES.
//
// MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN
// "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE,
// SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
//
#pragma once

#include <vector>

class DisjointSet
{
private:
	std::vector<int>	parent_;
	std::vector<int>	size_;

public:
	DisjointSet(const int n)
	{
		parent_.reserve(n);
		size_.reserve(n);

		for (int i=0; i<n; ++i)
		{
			parent_.push_back(i);
			size_.push_back(1);
		}
	}

	~DisjointSet() {}

	inline void remove(const int x) {
		if(parent_[x]!=x) {
			--size_[Find(x)];
			parent_[x]=x;
			size_[x]=1;
		}
	}

	inline int getSetSize(const int x) {
		return size_[Find(x)];
	}

	inline int Union(const int x, const int y)
	{
		const int xRoot = Find(x);
		const int yRoot = Find(y);

		if (xRoot == yRoot)
			return xRoot;

		const int xRootSize = size_[xRoot];
		const int yRootSize = size_[yRoot];

		if (xRootSize < yRootSize) {
			parent_[xRoot] = yRoot;
			size_[yRoot]+=size_[xRoot];
			return yRoot;
		} else {
			parent_[yRoot] = xRoot;
			size_[xRoot]+=size_[yRoot];
			return xRoot;
		}
	}

	inline int Find(const int x)
	{
		if (parent_[x] != x)
			parent_[x] = Find(parent_[x]);

		return parent_[x];
	}

private:
	DisjointSet();
	DisjointSet(const DisjointSet& rhs);
	const DisjointSet& operator=(const DisjointSet& rhs);
};