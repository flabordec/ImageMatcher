using System;
using System.Collections.Generic;
using System.Linq;

namespace MaguSoft.ImageMatcherCommon;

public class DisjointSet<T> where T : IEquatable<T>
{
    // Tracks the parent representative of each item
    private readonly Dictionary<T, T> _parents = new();

    // Tracks the depth of the tree to keep it balanced during Unions
    private readonly Dictionary<T, int> _ranks = new();

    public IEnumerable<ISet<T>> GetAllGroups()
    {
        var result =
            from item in _parents.Keys
            group item by Find(item) into g
            select (ISet<T>)g.ToHashSet();
        return result;
    }

    public void MakeSet(T item)
    {
        if (!_parents.ContainsKey(item))
        {
            _parents[item] = item;
            _ranks[item] = 0;
        }
    }

    public T Find(T item)
    {
        if (!_parents.ContainsKey(item))
        {
            throw new KeyNotFoundException($"The item '{item}' does not exist in the Disjoint Set.");
        }

        // Path Compression: recursively point all traversed nodes directly to the root
        if (!_parents[item].Equals(item))
        {
            _parents[item] = Find(_parents[item]);
        }

        return _parents[item];
    }

    public void Union(T item1, T item2)
    {
        T root1 = Find(item1);
        T root2 = Find(item2);

        if (root1.Equals(root2)) return;

        // Union by Rank: attach the shorter tree under the root of the taller tree
        if (_ranks[root1] < _ranks[root2])
        {
            _parents[root1] = root2;
        }
        else if (_ranks[root1] > _ranks[root2])
        {
            _parents[root2] = root1;
        }
        else
        {
            _parents[root2] = root1;
            _ranks[root1]++;
        }
    }
}