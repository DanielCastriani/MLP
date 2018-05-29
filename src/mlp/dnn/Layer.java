package mlp.dnn;

import java.util.ArrayList;
import java.util.Random;

public class Layer {

    private ArrayList<Double> values, act_values;
    private ArrayList<ArrayList<Double>> w;
    private ArrayList<Double> error;

    private int mode;

    public ArrayList<Double> getValues() {
        return values;
    }

    public ArrayList<Double> getAct_values() {
        return act_values;
    }

    public ArrayList<ArrayList<Double>> getW() {
        return w;
    }

    public void setMode(int mode) {
        this.mode = mode;
    }

    public ArrayList<Double> getError() {
        return error;
    }

    public Layer(int q_input, int q_output) {

        Random rd = new Random();
        rd.setSeed(10101010);

        values = new ArrayList<>();
        act_values = new ArrayList<>();
        error = new ArrayList<>();
        w = new ArrayList<>();

        for (int i = 0; i < q_input; i++) {
            values.add(new Double(0));
            act_values.add(new Double(0));
            error.add(new Double(0));
        }

        for (int i = 0; i < values.size(); i++) {
            ArrayList _w = new ArrayList();
            for (int j = 0; j < q_output; j++) {
                double val;
                if (rd.nextDouble() <= 0.5) {
                    val = rd.nextDouble() * -1;
                } else {
                    val = rd.nextDouble();
                }
                _w.add(val);
            }
            w.add(_w);
        }
    }

    public void inputInputLayer(ArrayList<Double> in) {
        for (int i = 0; i < in.size(); i++) {
            values.set(i, in.get(i));
            act_values.set(i, in.get(i));
        }
    }

    public void feed(Layer in) {
        for (int i = 0; i < getValues().size(); i++) {
            double sum = 0;
            for (int j = 0; j < in.getValues().size(); j++) {
                sum += in.getValues().get(j) * in.getW().get(j).get(i);
            }
            values.set(i, sum);
            act_values.set(i, f(sum));
        }
    }

    void newW(double learning_rate,ArrayList<Double> error) {
        for (int i = 0; i < w.size(); i++) {

            for (int j = 0; j < w.get(i).size(); j++) {
                double np = w.get(i).get(j) + learning_rate * error.get(j) * getAct_values().get(i);
                w.get(i).set(j, np);
            }
        }
    }

    //<editor-fold defaultstate="collapsed" desc="Activation Functions">
    public double f(double x) {
        double val = 0;
        switch (mode) {
            case 1:
                val = linear(x);
                break;
            case 2:
                val = logistic(x);
                break;
            case 3:
                val = tanh(x);
                break;
            default:
                val = tanh(x);
                break;
        }
        return val;
    }

    public double d_f(double x) {
        double val = 0;
        switch (mode) {
            case 1:
                val = d_linear(x);
                break;
            case 2:
                val = d_logistic(x);
                break;
            case 3:
                val = d_tanh(x);
                break;
            default:
                val = d_tanh(x);
                break;
        }
        return val;
    }

    public double linear(double x) {
        return x / 10;
    }

    public double d_linear(double x) {
        return 0.1;
    }

    public double logistic(double x) {
        return 1.0 / (1 + Math.pow(Math.E, -x));
    }

    public double d_logistic(double x) {
        return logistic(x) * (1 - logistic(x));
    }

    public double tanh(double x) {
        double e = Math.pow(Math.E, -2 * x);
        double a = 1 - e;
        double b = 1 + e;
        return a / b;
    }

    public double d_tanh(double x) {
        return 1 - Math.pow(tanh(x), 2);
    }
    //</editor-fold>

}
