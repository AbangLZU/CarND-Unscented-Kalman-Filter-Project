#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 0.9;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.4;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
    TODO:

    Complete the initialization. See ukf.h for other member properties.

    Hint: one or more values initialized above might be wildly off...
    */
    is_initialized_ = false;
    n_x_ = 5;
    n_aug_ = 7;
    lambda_ = 3 - n_aug_;
    use_laser_ = true;
    use_radar_ = true;

    P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    x_.fill(0.0);
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    weights_ = VectorXd(2 * n_aug_ + 1);
    weights_[0] = lambda_/ (lambda_ + n_aug_);
    for(int i=1; i < (2*n_aug_+1); i++){
        weights_[i] = 1/(2 * (lambda_ + n_aug_));
    }

    R_laser_ = MatrixXd(2, 2);
    R_laser_ << std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;

    R_radar_ = MatrixXd(3, 3);
    R_radar_ << std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0, std_radrd_*std_radrd_;

    H_laser_ = MatrixXd(2, n_x_);
    H_laser_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0;

    NIS_laser_ = MatrixXd(2, 2);
    NIS_radar_ = MatrixXd(3, 3);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Make sure you switch between lidar and radar
    measurements.
    */
    if (!is_initialized_) {
        // first measurement
        x_ << 1, 1, 1, 1, 0.1;

        if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            x_[0] = meas_package.raw_measurements_[0];
            x_[1] = meas_package.raw_measurements_[1];
        } else {
            float rho = meas_package.raw_measurements_[0];
            float phi = meas_package.raw_measurements_[1];
            float rho_dot = meas_package.raw_measurements_[2];
            x_[0] = rho * cos(phi);
            x_[1] = rho * sin(phi);
        }
        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }
    cout<<x_<<endl;
    double delta_t =(meas_package.timestamp_ - time_us_) /  1000000.0;
    cout<<"delta_t"<<delta_t<<endl;
    time_us_ = meas_package.timestamp_;
    Prediction(delta_t);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        UpdateLidar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    TODO:

    Complete this function! Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    Xsig_aug.fill(0.0);
    AugmentedSigmaPoints(&Xsig_aug);
    SigmaPointPrediction(Xsig_aug, delta_t);
    cout<<"Xsig_pred_:"<<Xsig_pred_<<endl;
    PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */
    VectorXd z = meas_package.raw_measurements_;
    VectorXd y = z - H_laser_ * x_;
    MatrixXd S = H_laser_ * P_ * H_laser_.transpose() + R_laser_;
    MatrixXd K = P_ * H_laser_.transpose() * S.inverse();

    x_ += K * y;
    P_ = (MatrixXd::Identity(n_x_, n_x_) - K * H_laser_) * P_;

    NIS_laser_ = y.transpose() * S.inverse() * y;
    cout << "NIS_laser:"<<NIS_laser_<<endl;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
    */
    int n_z = 3;
    VectorXd z = VectorXd(n_z);
    z << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1],
            meas_package.raw_measurements_[2];

    VectorXd z_pred = VectorXd(3);
    MatrixXd S_out = MatrixXd(3, 3);
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    PredictRadarMeasurement(&z_pred, &S_out, &Zsig);

    cout << "z_pred" << z_pred<<endl;
    cout << "S"<< S_out <<endl;
    cout << "Zsig" <<Zsig<<endl;

    UpdateState(z, z_pred, S_out, Zsig);
    VectorXd z_diff = z - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    NIS_radar_ = z_diff.transpose() * S_out.inverse() * z_diff;
    cout << "NIS_radar:"<<NIS_radar_<<endl;
}

void UKF::AugmentedSigmaPoints(MatrixXd *Xsig_out) {

    //create augmented mean vector
    VectorXd x_aug = VectorXd(7);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    //create augmented covariance matrix
    //create square root matrix
    //create augmented sigma points
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5,5) = std_a_*std_a_;
    P_aug(6,6) = std_yawdd_*std_yawdd_;

    MatrixXd A = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i< n_aug_; i++)
    {
        Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * A.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A.col(i);
    }
    cout<<"Xsig_aug:"<<Xsig_aug<<endl;

    //write result
    *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd &Xsig_aug, double delta_t) {

    //predict sigma points
    //avoid division by zero
    //write predicted sigma points into right column
    for (int i = 0; i< 2*n_aug_+1; i++) {
        //extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a * delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }
}

void UKF::PredictMeanAndCovariance() {
    x_.fill(0.0);
    for(int i=0; i<2*n_aug_+1; i++){
        x_ = x_+ weights_[i] * Xsig_pred_.col(i);
    }

    cout<<"x_"<<x_<<endl;

    P_.fill(0.0);
    for(int i=0;  i<2*n_aug_+1; i++){
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        while (x_diff[3]> M_PI)
            x_diff[3] -= 2.*M_PI;
        while (x_diff[3] <-M_PI)
            x_diff[3]+=2.*M_PI;
        P_ = P_ + weights_[i] * x_diff * x_diff.transpose();
    }
    cout << "P_"<< P_<<endl;
}

void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out) {

    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);

    //transform sigma points into measurement space
    //calculate mean predicted measurement
    //calculate measurement covariance matrix S
    for(int i=0; i < 2*n_aug_+1; i++){
        float px = Xsig_pred_.col(i)[0];
        float py = Xsig_pred_.col(i)[1];
        float v = Xsig_pred_.col(i)[2];
        float psi = Xsig_pred_.col(i)[3];
        float psi_dot = Xsig_pred_.col(i)[4];

        float temp = px * px + py * py;
        if(fabs(temp) < 0.0001){
            temp = 0.0001;
        }
        float rho = sqrt(temp);
        float phi = atan2(py, px);
        float rho_dot =(px * cos(psi) * v + py * sin(psi) * v)/rho;

        VectorXd temp1 = VectorXd(3);
        temp1 << rho, phi, rho_dot;
        Zsig.col(i) = temp1;
    }
    cout<<"Zsig:"<<Zsig<<endl;

    for(int i=0; i < 2*n_aug_+1; i++){
        z_pred = z_pred + weights_[i] * Zsig.col(i);
    }

    for(int i=0; i < 2*n_aug_+1; i++){
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
        S = S + weights_[i] * (Zsig.col(i) - z_pred) * (Zsig.col(i) - z_pred).transpose();
    }
    S = S + R_radar_;

    //write result
    *z_out = z_pred;
    *S_out = S;
    *Zsig_out = Zsig;
}


void UKF::UpdateState(VectorXd &z, VectorXd &z_pred, MatrixXd &S, MatrixXd &Zsig) {

    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
    //calculate Kalman gain K;
    //update state mean and covariance matrix

    for(int i=0; i < 2*n_aug_+1; i++){
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        Tc = Tc + weights_[i] * x_diff * z_diff.transpose();
    }

    MatrixXd K = MatrixXd(5, 3);
    K = Tc * S.inverse();

    VectorXd y = z - z_pred;
    //angle normalization
    while (y(1)> M_PI) y(1)-=2.*M_PI;
    while (y(1)<-M_PI) y(1)+=2.*M_PI;

    x_ = x_ + K * y;
    P_ = P_ - K * S * K.transpose();
}

