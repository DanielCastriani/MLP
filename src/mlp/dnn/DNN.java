package mlp.dnn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.stage.FileChooser;

public class DNN {

    //dataset
    private ArrayList<Double> classes;
    private ArrayList<Reg> x_train;
    private ArrayList<Reg> x_test;

    private int q_features;
    private int q_output;
    private int q_hidden_neurons;
    private int q_hidden_layers;
    private int mode;

    private ArrayList<Layer> layers;

    private int it;
    private double error;
    private double l_rate;

    private double erroGeral;

    private int[][] mc;

    private int acertos;

    public DNN() {

    }
    //<editor-fold defaultstate="collapsed" desc="Getters & Setters">

    public int getQ_features() {
        return q_features;
    }

    public int getQ_output() {
        return q_output;
    }

    public int getQ_hidden_neurons() {
        return q_hidden_neurons;
    }

    public ArrayList<Reg> getX_train() {
        return x_train;
    }

    public ArrayList<Reg> getX_test() {
        return x_test;
    }

    public void setMode(int mode) {
        this.mode = mode;
    }

    public void setIt(int it) {
        this.it = it;
    }

    public void setError(double error) {
        this.error = error;
    }

    public void setL_rate(double l_rate) {
        this.l_rate = l_rate;
    }

    public void setQ_hidden_neurons(int q_hidden_neurons) {
        this.q_hidden_neurons = q_hidden_neurons;
    }

    public void setQ_hidden_layers(int q_hidden_layers) {
        this.q_hidden_layers = q_hidden_layers;
    }

    public int getAcertos() {
        return acertos;
    }

    public ArrayList<Double> getClasses() {
        return classes;
    }

    public int[][] getMc() {
        return mc;
    }

    //</editor-fold>    
    //<editor-fold defaultstate="collapsed" desc="INIT">
    public void normalize(ArrayList<Reg> x) {
        Reg r = x.get(0);

        Reg max = new Reg();
        max.setX(new ArrayList());

        for (int i = 0; i < r.getX().size(); i++) {
            max.getX().add(r.getX().get(i));
        }

        for (Reg l : x) {
            for (int i = 0; i < l.getX().size(); i++) {
                if ((double) l.getX().get(i) > (double) max.getX().get(i)) {
                    max.getX().set(i, l.getX().get(i));
                }
            }
        }

        for (Reg l : x) {
            for (int i = 0; i < l.getX().size(); i++) {
                double norma = (double) l.getX().get(i) / (double) max.getX().get(i);

                l.getX().set(i, norma);
            }
        }
    }

    public void init_layers() {
        layers = new ArrayList<>();
        Layer input = new Layer(q_features, q_hidden_neurons);
        input.setMode(mode);
        layers.add(input);
        for (int i = 0; i < q_hidden_layers - 1; i++) {
            Layer h = new Layer(q_hidden_neurons, q_hidden_neurons);
            h.setMode(mode);
            layers.add(h);
        }

        Layer h = new Layer(q_hidden_neurons, q_output);
        h.setMode(mode);
        layers.add(h);

        Layer output = new Layer(q_output, q_output);
        output.setMode(mode);
        layers.add(output);
    }

    public boolean init(int hidden_layers) {
        x_train = load(true);
        if (x_train != null) {
            x_test = load(false);
            if (x_test != null) {
                Collections.shuffle(x_train);
                Collections.shuffle(x_test);
                this.q_hidden_layers = hidden_layers;
                normalize(x_train);
                normalize(x_test);

                mc = new int[x_test.size()][x_test.size()];
                return true;
            }
        }
        return false;
    }

    private ArrayList<Reg> load(boolean train) {
        FileChooser fc = new FileChooser();
        if (train) {
            fc.setTitle("Abra um arquivo de treino");
        } else {
            fc.setTitle("Abra um arquivo de teste");
        }
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV", "*.csv"));
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("All", "*"));
        fc.setInitialDirectory(new File("."));
        File f = fc.showOpenDialog(null);

        try {
            if (f != null) {

                ArrayList<Reg> x = new ArrayList<Reg>();
                if (train) {
                    classes = new ArrayList<>();
                }

                BufferedReader br = new BufferedReader(new FileReader(f.getAbsoluteFile()));
                String line = br.readLine();
                String[] info = line.split(",");

                while ((line = br.readLine()) != null) {
                    info = line.split(",");

                    ArrayList<Double> list_atrib = new ArrayList<>();
                    for (int i = 0; i < info.length - 1; i++) {
                        String s = info[i];
                        list_atrib.add(Double.parseDouble(s));
                    }
                    Double y_true = Double.parseDouble(info[info.length - 1]);
                    Reg r = new Reg(list_atrib, -1.0, y_true);
                    x.add(r);

                    if (train) {
                        if (!classes.contains(y_true)) {
                            classes.add(y_true);
                        }
                    }
                }
                if (train) {
                    q_features = x.get(0).getX().size();
                    q_output = classes.size();
                    q_hidden_neurons = (int) Math.sqrt((double) q_features * (double) q_output);
                }
                return x;
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DNN.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(DNN.class.getName()).log(Level.SEVERE, null, ex);
        }

        return null;
    }
    //</editor-fold>

    public void train() {
        System.out.println("Treino Iniciou!!!!");
        System.out.println("-------------------------------------------");

        int it = 0;
        double tx = l_rate;
        double erroMedio = Double.MAX_VALUE;
        while (it < this.it && erroMedio > error) {
            erroMedio = 0;
            System.out.println("Epoch:" + (it + 1));
            for (int i = 0; i < x_train.size(); i++) {
                Layer input = layers.get(0);
                input.inputInputLayer(x_train.get(i).getX());
                feedForward(i);
                outputError(i);

                for (int j = layers.size() - 2; j >= 0; j--) {
                    layerError(layers.get(j), layers.get(j + 1));
                }

                for (int j = layers.size() - 2; j >= 0; j--) {
                    Layer a = layers.get(j);
                    Layer b = layers.get(j + 1);
                    a.newW(tx, b.getError());
                }

                erroMedio += erroGeral;
            }
            it++;
            erroMedio /= x_test.size();
            double l = 1 - erroMedio;
            tx = l_rate * erroMedio;
            System.out.println("Erro Médio:" + erroMedio);
            System.out.println("TX:" + tx);
            teste();
            System.out.println("-------------------------------------------");

        }
    }

    public void teste() {
        Layer output = layers.get(layers.size() - 1);

        for (int i = 0; i < mc.length; i++) {
            for (int j = 0; j < mc[i].length; j++) {
                mc[i][j] = 0;
            }
        }

        for (int i = 0; i < x_test.size(); i++) {
            Layer input = layers.get(0);
            input.inputInputLayer(x_test.get(i).getX());
            feedForward(i);

            int index = 0;
            double max = output.getAct_values().get(index);

            for (int j = 1; j < output.getAct_values().size(); j++) {
                if (output.getAct_values().get(j) > max) {
                    max = output.getAct_values().get(j);
                    index = j;
                }
            }

            x_test.get(i).setY(classes.get(index));

            int index_true = classes.indexOf(x_test.get(i).getY_true());
            mc[index][index_true]++;
        }
        acertos = 0;

        for (int i = 0; i < x_test.size(); i++) {
            acertos += mc[i][i];
        }
        System.out.println("Acertos:" + acertos + "/" + x_test.size());
    }

    private void feedForward(int i) {
        layers.get(1).feed(layers.get(0));
        for (int j = 1; j < layers.size() - 1; j++) {
            Layer a = layers.get(j);
            Layer b = layers.get(j + 1);
            b.feed(a);
        }
    }

    private void outputError(int p) {
        Layer output = layers.get(layers.size() - 1);

        erroGeral = 0;
        double erro = 0;

        double y_true;
        int index = classes.indexOf(x_train.get(p).getY_true());

        for (int i = 0; i < output.getAct_values().size(); i++) {
            if (i == index) {
                y_true = 1;
            } else {
                y_true = 0;
            }

            double y = output.getAct_values().get(i);
            erro = (y_true - y) * output.d_f(output.getValues().get(i));
            output.getError().set(i, erro);
            erroGeral += Math.pow(erro, 2);
        }
        erroGeral *= 0.5;
    }

    private void layerError(Layer a, Layer b) {
        double error;
        for (int i = 0; i < a.getValues().size(); i++) {
            error = 0;
            for (int j = 0; j < b.getValues().size(); j++) {
                error += b.getError().get(j) * a.getW().get(i).get(j) * a.d_f(a.getValues().get(i));
            }
            a.getError().set(i, error);
        }
    }

}
