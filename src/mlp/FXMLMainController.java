package mlp;

import java.net.URL;
import java.util.ResourceBundle;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.RadioButton;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleGroup;
import javafx.scene.control.cell.PropertyValueFactory;
import javax.swing.JOptionPane;
import mlp.dnn.DNN;

public class FXMLMainController implements Initializable {

    @FXML
    private TextField tfEntrada;
    @FXML
    private TextField tfSaida;
    @FXML
    private TextField tfOculta;
    @FXML
    private TextField tfErro;
    @FXML
    private TextField tfIteracao;
    @FXML
    private TextField tfN;
    @FXML
    private RadioButton rbLinear;
    @FXML
    private RadioButton rbLogistica;
    @FXML
    private RadioButton rbTanh;
    @FXML
    private Label lbStatus;
    @FXML
    private TableView<Object> tvTrain;
    @FXML
    private TableColumn<?, ?> tvTrainFeatures;
    @FXML
    private TableColumn<?, ?> tvTrainY;
    @FXML
    private TableColumn<?, ?> tvTrainYTRUE;
    @FXML
    private TableView<Object> tvTest;
    @FXML
    private TableColumn<?, ?> tvTestFeatures;
    @FXML
    private TableColumn<?, ?> tvTestY;
    @FXML
    private TableColumn<?, ?> tvTestYTRUE;

    private DNN dnn;
    @FXML
    private Button btAbrirDataset;
    @FXML
    private Button btTreino;
    @FXML
    private Button btTeste;
    @FXML
    private TextField tfQtdOculta;

    @Override
    public void initialize(URL url, ResourceBundle rb) {
        ToggleGroup tg = new ToggleGroup();

        rbLinear.setToggleGroup(tg);
        rbLogistica.setToggleGroup(tg);
        rbTanh.setToggleGroup(tg);

        rbTanh.setSelected(true);

        disableBtns(true);
        tvTrainFeatures.setCellValueFactory(new PropertyValueFactory<>("x"));
        tvTrainY.setCellValueFactory(new PropertyValueFactory<>("y"));
        tvTrainYTRUE.setCellValueFactory(new PropertyValueFactory<>("y_true"));

        tvTestFeatures.setCellValueFactory(new PropertyValueFactory<>("x"));
        tvTestY.setCellValueFactory(new PropertyValueFactory<>("y"));
        tvTestYTRUE.setCellValueFactory(new PropertyValueFactory<>("y_true"));

        dnn = new DNN();

        tfErro.setText("0.01");
        tfIteracao.setText("10");
        tfN.setText("0.01");
        tfQtdOculta.setText("2");
    }

   
    private void disableBtns(boolean b) {
        btTeste.setDisable(b);
        btTreino.setDisable(b);
    }

    @FXML
    private void OnAction_AbrirDataset(ActionEvent event) {
        int qtd_hidden = 2;
        if (dnn.init(qtd_hidden)) {
            disableBtns(false);
            updateTabela();
            tfOculta.setText(dnn.getQ_hidden_neurons() + "");
        } else {
            disableBtns(true);
        }
    }

    private boolean init_parametros() {
        try {

            double tx = Double.parseDouble(tfN.getText());
            double erro = Double.parseDouble(tfErro.getText());
            int it = Integer.parseInt(tfIteracao.getText());
            int qtdHiddenN = Integer.parseInt(tfOculta.getText());
            int qtdHiddenL = Integer.parseInt(tfQtdOculta.getText());

            if (qtdHiddenL <= 0) {

                tfQtdOculta.setStyle("-fx-background-color:red");
                return false;
            } else {

                tfQtdOculta.setStyle("");
            }

            if (qtdHiddenN <= 0) {
                tfOculta.setStyle("-fx-background-color:red");
                return false;
            } else {
                tfOculta.setStyle("");
            }

            dnn.setL_rate(tx);
            dnn.setError(erro);
            dnn.setIt(it);
            dnn.setQ_hidden_neurons(qtdHiddenN);
            dnn.setQ_hidden_layers(qtdHiddenL);

            dnn.setMode(rbLinear.isSelected() ? 1 : (rbLogistica.isSelected() ? 2 : 3));

            dnn.init_layers();
            return true;
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(null, "Preencha os parametros corretamente");
        }
        return false;
    }

    @FXML
    private void OnAction_Treino(ActionEvent event) {
        if (init_parametros()) {
            disableBtns(true);
            dnn.train();
            updateTabela();
            lbStatus.setText(dnn.getAcertos() + "");
            disableBtns(false);
        }
    }

    @FXML
    private void OnAction_Teste(ActionEvent event) {
        dnn.teste();
    }

    private void updateTabela() {
        try {
            if (dnn.getX_test() != null) {
                tvTest.setItems(FXCollections.observableArrayList(dnn.getX_test()));
            }
            if (dnn.getX_train() != null) {
                tvTrain.setItems(FXCollections.observableArrayList(dnn.getX_train()));
            }
        } catch (Exception ex) {
            System.out.println("updateTabela:" + ex);
        }
    }

}
