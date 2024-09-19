import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

class Point {
    double[] feature;
    String label;
    public Point(double[] _feature, String _label) {
        this.feature = _feature;
        this.label = _label;
    }
}

class Neighboor {
    double distance; 
    String label;
    public Neighboor(double _distance, String _label) {
        this.distance = _distance;
        this.label = _label;
    }
}

public class KNNParallel {
    public static int N = 2000000;
    public static int H = 2;
    public static int d = (int) N / H;
    public List<Neighboor> neighbors = Collections.synchronizedList(new ArrayList<>());
    public List<Point> Data = new ArrayList<>();
    
    public static double EuclideanDistance(double[] p1, double[] p2) {
        double sum = 0;
        for (int i = 0; i < p1.length; ++i) {
            sum += Math.pow(p1[i] - p2[i], 2);
        }
        return Math.sqrt(sum);
    }
    
    public String classify(double[] newp, int k) {
        Thread threads[] = new Thread[40];
        for (int i = 0; i < H; ++i) {
            int begin = i * d;
            int end = (i + 1 == H) ? N : (i + 1) * d;
            threads[i] = new Procces(begin, end, i, newp);
            threads[i].start();
        }
        for (int i = 0; i < H; ++i) {
            try {
                threads[i].join();
            } catch (InterruptedException ex) {
                System.out.println("error " + ex);
            }
        }
        Collections.sort(neighbors, Comparator.comparingDouble(n -> n.distance));
        int cntA = 0;
        for (int i = 0; i < k; ++i) {
            if (neighbors.get(i).label.equals("A")) cntA++;
            else if (neighbors.get(i).label.equals("B")) cntA--;
        }
        return (cntA > 0) ? "A" : "B";
    }
    public class Procces extends Thread {
        public int begin, end, id;
        public double[] newpoint;
        Procces(int _begin, int _end, int _id, double[] _newpoint) {
            this.begin = _begin;
            this.end = _end;
            this.id = _id;
            this.newpoint = _newpoint;
        }
        public void run() {
            for (int i = begin; i < end; ++i) {
                double dis = EuclideanDistance(Data.get(i).feature, newpoint);
                neighbors.add(new Neighboor(dis, Data.get(i).label)); 
            }
            System.out.println("id: " + id + ", begin: " + begin + ", end: " + end);
        }
    }
    public static void main (String[] args) {
        KNNParallel knn = new KNNParallel();
        String data = "data.txt";
        List<Point> training = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(data))){
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split("\\s+");
                double e1 = Double.parseDouble(values[0]);
                double e2 = Double.parseDouble(values[1]);
                String s = values[2];
                training.add(new Point(new double[] {e1, e2}, s));
            }
        } catch (IOException e) {
            System.out.println("erro " + e.getMessage());
        }
        double[] newp = {2.5, 3.5};
        knn.Data = training;
        int k = 3;
        long startTime = System.currentTimeMillis();
        String result = knn.classify(newp, k);
        long endTime = System.currentTimeMillis();
        long ans = endTime - startTime;
        System.out.println("the class belong the newp is : " + result);
        System.out.println("the execution time: " + ans); 
    }
}
