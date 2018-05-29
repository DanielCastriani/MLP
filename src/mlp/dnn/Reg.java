package mlp.dnn;

import java.util.ArrayList;

public class Reg {
    private ArrayList<Double> x;
    private Double y;
    private Double y_true;

    public Reg() {
    }

    public Reg(ArrayList<Double> x, Double y, Double y_true) {
        this.x = x;
        this.y = y;
        this.y_true = y_true;
    }

    public ArrayList<Double> getX() {
        return x;
    }

    public void setX(ArrayList<Double> x) {
        this.x = x;
    }

    public Double getY() {
        return y;
    }

    public void setY(Double y) {
        this.y = y;
    }

    public Double getY_true() {
        return y_true;
    }

    public void setY_true(Double y_true) {
        this.y_true = y_true;
    }

    
   
}
