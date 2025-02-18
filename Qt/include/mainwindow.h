#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);  // Constructor
    ~MainWindow();  // Destructor

private:
    // Add private members or methods if needed
};

#endif // MAINWINDOW_H