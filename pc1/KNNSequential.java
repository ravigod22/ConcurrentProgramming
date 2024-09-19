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

public class KNNSequential {
    public static double EuclideanDistance(double[] p1, double[] p2) {
        double sum = 0;
        for (int i = 0; i < p1.length; ++i) {
            sum += Math.pow(p1[i] - p2[i], 2);
        }
        return Math.sqrt(sum);
    }
    
    public static String classify(List<Point> Data, double[] newp, int k) {
        List<Neighboor> neighbors = new ArrayList<>();
        for (Point p : Data) {
            double dis = EuclideanDistance(p.feature, newp);
            neighbors.add(new Neighboor(dis, p.label));
        }
        Collections.sort(neighbors, Comparator.comparingDouble(n -> n.distance));
        int cntA = 0;
        for (int i = 0; i < k; ++i) {
            if (neighbors.get(i).label.equals("A")) cntA++;
            else if (neighbors.get(i).label.equals("B")) cntA--;
        }
        return (cntA > 0) ? "A" : "B";
    }
    public static void main (String[] args) {
        // leer data
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

        int k = 3;
        String result = classify(training, newp, k);
        System.out.println("the class belong the newp is : " + result);
    }
}
