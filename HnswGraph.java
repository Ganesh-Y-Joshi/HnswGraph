package org.joshi.gyj.hnsw;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.SortedSet;
import java.util.concurrent.ConcurrentSkipListSet;

public class HnswGraph {

    public record VectorNode(List<Double> vector, Double cost) implements Comparable<VectorNode> {

        @Override
            public int compareTo(VectorNode o) {
                return this.cost.compareTo(o.cost);
            }

            @Override
            public String toString() {
                return "VectorNode{" +
                        "vector=" + vector +
                        ", cost=" + cost +
                        '}';
            }
        }
    
    public static class HnsNode {
        private final SortedSet<VectorNode> hnsSet;
        private final int maxSize;
        private double tcost;

        public HnsNode(int maxSize) {
            this.maxSize = maxSize;
            hnsSet = new ConcurrentSkipListSet<>();
            tcost = 0;
        }
        
        public Optional<VectorNode> addNode(List<Double> vector) {
            if (vector.isEmpty()) {
                return Optional.of(new VectorNode(null, -1.));
            }
            
            if (hnsSet.isEmpty()) {
                hnsSet.add(new VectorNode(vector, 0.));
                return Optional.empty();
            }
            
            else {
                var baseEntry = hnsSet.first();
                var cost = calculateCost(vector, baseEntry.vector());

                if (cost == -1) {
                    throw new IllegalArgumentException("Please pass the vectors with the same dim");
                }
                
                if (hnsSet.size() == this.maxSize) {
                    var last = hnsSet.last();
                    if (cost > last.cost()) {
                        return Optional.of(new VectorNode(vector, cost));
                    } else {
                        // Remove the last entry and remove
                        tcost -= hnsSet.last().cost();
                        hnsSet.add(new VectorNode(vector, cost));
                        tcost += cost;
                        hnsSet.removeLast();
                        return Optional.of(last);
                    }
                } else {
                    hnsSet.add(new VectorNode(vector, cost));
                    return Optional.empty();
                }
            }
        }

        public Optional<VectorNode> findMinCost(List<Double> vector) {
            if (vector.isEmpty()) {
                return Optional.empty();
            }

            double minCost = Integer.MAX_VALUE;
            List<Double> cNode = new ArrayList<>();
            for (var node: hnsSet) {
                var cCost = calculateCost(node.vector, vector);
                if (cCost < minCost) {
                    minCost = cCost;
                    cNode = node.vector;
                }
            }
            return Optional.of(new VectorNode(cNode, minCost));
        }
        
        public double calculateCost(List<Double> v1, List<Double> v2) {
            if (v1.size() != v2.size()) {
                return -1.;
            }
            
            double cost = 0.;
            for (int i = 0; i < v1.size(); i++) {
                cost += euclideanF(
                        v1.get(i), v2.get(i)
                );
            }
            return cost;
        }
        
        public double euclideanF(double v1, double v2) {
            return Math.sqrt(Math.abs(Math.pow((v1 - v2), 2)));
        }

        public SortedSet<VectorNode> getHnsSet() {
            return hnsSet;
        }

        public int getMaxSize() {
            return maxSize;
        }

        public boolean isFull() {
            return hnsSet.size() == maxSize;
        }
    }

    private final int maxSize;
    private final ArrayList<HnswGraph.HnsNode> graph;
//    private final SortedSet<HnswGraph.HnsNode> grSet;
    public HnswGraph(int maxSize) {
        if (maxSize <= 0) {
            maxSize = 100;
        }

        this.maxSize = maxSize;
        this.graph = new ArrayList<>();
//        this.grSet = new ConcurrentSkipListSet<>();
    }

    public void addNode(List<Double> vector) {
        if (this.graph.isEmpty()) {
            var node = new HnsNode(maxSize);
            graph.add(node);
            node.addNode(vector);
            return;
        }

        if (areAllNodeFilled()) {
            Optional<VectorNode> n = Optional.empty();
            for (var node: this.graph) {
                int i = 0;
                // In this case if we add the node it will always return the data
                n = node.addNode(vector);
            }

            graph.add(new HnsNode(maxSize));
            if (n.isPresent())
                graph.getLast().addNode(n.get().vector());
            return;
        }

        for (var node: this.graph) {
            int i = 0;
            // In this case if we add the node it will always return the data
            var n = node.addNode(vector);
            if (n.isEmpty()) {
                return;
            }
        }
    }

    public Optional<VectorNode> findMin(List<Double> vector) {
        double minCost = Integer.MAX_VALUE;
        var minIdx = -1;
        for (int i = 0; i < graph.size(); i++) {
            var cost = graph.get(i).calculateCost(graph.get(i).hnsSet.first().vector, vector);
            if (cost < minCost) {
                minCost = cost;
                minIdx = i;
            }
        }

        return graph.get(minIdx).findMinCost(vector);
    }

    private boolean areAllNodeFilled() {
        return graph.getLast().isFull();
    }

    public static void main(String[] args) {
        HnswGraph hnswGraph = new HnswGraph(10);

        // Test vectors
        List<Double> vector1 = Arrays.asList(1.0, 2.0, 3.0);
        List<Double> vector2 = Arrays.asList(4.0, 5.0, 6.0);
        List<Double> vector3 = Arrays.asList(7.0, 8.0, 9.0);
        List<Double> vector4 = Arrays.asList(0.0, 0.0, 0.0);

        // Add test vectors to the graph
        hnswGraph.addNode(vector1);
        hnswGraph.addNode(vector2);
        hnswGraph.addNode(vector3);
        hnswGraph.addNode(vector4);

        // Test query vectors
        List<Double> queryVector1 = Arrays.asList(0.9, 2.0, 2.8);
        List<Double> queryVector2 = Arrays.asList(5.0, 5.2, 6.3);
        List<Double> queryVector3 = Arrays.asList(7.5, 8.5, 8.9);

        // Perform queries
        System.out.println("Nearest neighbor for queryVector1: " + hnswGraph.findMin(queryVector1));
        System.out.println("Nearest neighbor for queryVector2: " + hnswGraph.findMin(queryVector2));
        System.out.println("Nearest neighbor for queryVector3: " + hnswGraph.findMin(queryVector3));
    }
}
