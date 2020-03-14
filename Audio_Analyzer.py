#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0303
# ******************************************************
#         @author: Haifeng CHEN - optical.dlz@gmail.com
# @date (created): 2016-05-12 09:24
#           @file: Audio_Analyzer.py
#          @brief: Draw powerspectrum of collect audio info
#       @internal:
#        revision: 15
#   last modified: 2020-03-14 18:30:47
# *****************************************************

from __future__ import print_function  # (at top of module)

import os
import sys
import threading
import pyaudio
import numpy as np
import guiqwt.plot as plt
from guiqwt.builder import make
from guiqwt.annotations import AnnotatedPoint
from guiqwt.styles import AnnotationParam
from qtpy import QtCore, QtWidgets, QtGui
from detect_peaks import detect_peaks
from recorder import listen_for_signal, get_noise_level
from analyzer import calc_powerspectrum, cubic_maximize

IS_PY2 = sys.version[0] == '2'
if IS_PY2:
    import Queue as queue
else:
    import queue as queue

REV = 15

AUDIO_RATE = (8000, 11205, 16000, 22050, 32000, 44100, 48000,
              88200, 96000, 176400, 192000, 352800, 384000)

WINDOWS_SIZE = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)

WFUNC_NAMES = (u'Rectangular window', u'Triangular window',
               u'Blackman window', u'Blackman-Harris window',
               u'Hamming window', u'Hanning window', u'Bartlett window',
               u'Gaussian (σ=2.5) window', u'Gaussian (σ=3.5) window',
               u'Gaussian (σ=4.5) window')

WFUNC_TYPES = (u'boxcar', u'triang', u'blackman', u'blackmanharris',
               u'hamming', u'hann', u'bartlett',
               (u'gaussian', 2.5), (u'gaussian', 3.5), (u'gaussian', 4.5))


class Listener(QtCore.QThread):
    """ Listens to microphone, extracts valid signal from it
    and puts it to given queue for further analysis """
    # signal
    # status = QtCore.pyqtSignal(str, int)
    status = QtCore.Signal(str, int)

    def __init__(self, settings, queue, parent=None):
        super(Listener, self).__init__(parent)
        self.settings = settings
        self.queue = queue
        self.stop_ev = threading.Event()

    def quit(self):
        self.stop_ev.set()
        super(Listener, self).quit()

    def start(self):
        self.stop_ev.clear()
        super(Listener, self).start()

    def status_cb(self, what, val):
        self.status.emit(what, val)

    def run(self):
        rate = self.settings.value('rate', defaultValue=6, type=int)
        threshold = self.settings.value('threshold',
                                        defaultValue=2500, type=int)
        silence_limit = self.settings.value('silence_limit',
                                            defaultValue=1.0, type=float)
        prev_audio = self.settings.value('prev_audio',
                                         defaultValue=0.5, type=float)
        max_len = self.settings.value('max_len',
                                      defaultValue=5.0, type=float)
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1, rate=AUDIO_RATE[rate],
                        input=True, frames_per_buffer=1024)
        try:
            while not self.stop_ev.is_set():
                audio = listen_for_signal(stream, event=self.stop_ev,
                                          cb_fun=self.status_cb,
                                          max_len=max_len,
                                          rate=AUDIO_RATE[rate],
                                          threshold=threshold,
                                          silence_limit=silence_limit,
                                          prev_audio=prev_audio)
                if not self.stop_ev.is_set():
                    data = np.frombuffer(bytes.join(b'', audio), dtype=np.int16)
                    self.queue.put(data)
        except:
            print('Unexpected error: ',
                  sys.exc_info()[0], sys.exc_info()[1])

        self.queue.put(None)
        stream.close()
        p.terminate()
        return


class Analyzer(QtCore.QThread):
    """ Get recorded data from Listener and do the analysis """
    # signal
    # analyzed = QtCore.pyqtSignal()
    analyzed = QtCore.Signal()

    def __init__(self, settings, in_q, out_q, parent=None):
        super(Analyzer, self).__init__(parent)
        self.settings = settings
        self.au_q = in_q
        self.res_q = out_q

    def run(self):
        # get parameters
        while True:
            # get audio content in numpy array format
            audio = self.au_q.get()
            if audio is None:
                return
            self.analyze(audio)
            self.au_q.task_done()
            self.analyzed.emit()

    def analyze(self, data):
        wfunc = self.settings.value('window_func', defaultValue=5, type=int)
        wsize = self.settings.value('window_size', defaultValue=4, type=int)
        ps = calc_powerspectrum(data, WFUNC_TYPES[wfunc],
                                wsize=WINDOWS_SIZE[wsize])
        self.res_q.put((data, ps))


def loadQIcon(ico_name):
    """Load ico file as QT icon object"""
    icon = QtGui.QIcon()
    # determine if application is a script file or frozen exe
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    elif __file__:
        application_path = os.path.dirname(__file__)
    icon_path = os.path.join(application_path, ico_name)
    icon.addPixmap(QtGui.QPixmap(icon_path),
                   QtGui.QIcon.Normal, QtGui.QIcon.Off)
    return icon


def alignComboBoxText(combo, QT_Align):
    """
        Use setAlignment to align text in QComboBox
    """
    combo.setEditable(True)
    combo.lineEdit().setReadOnly(True)
    combo.lineEdit().setAlignment(QT_Align)
    for i in range(combo.count()):
        combo.setItemData(i, QT_Align, QtCore.Qt.TextAlignmentRole)


class FrequencyAnalysisWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(FrequencyAnalysisWidget, self).__init__(parent)
        self.setMinimumSize(320, 200)
        # guiqwt related attributes
        self.plot = None
        self.curve_item = None
        self.vx = None
        self.vy = None
        self.peak_item = None
        self.x = None
        self.y = None

    def setup_widget(self, title):
        self.plot = plt.CurvePlot(self, title=title,
                                  xunit='Hz', yunit='dB')
        self.curve_item = make.curve([], [])
        # set color to magenta
        color = QtGui.QColor(QtCore.Qt.magenta)
        color.setAlpha(150)
        self.curve_item.setPen(QtGui.QPen(color))
        self.curve_item.setBrush(QtGui.QBrush(color))
        self.plot.add_item(self.curve_item)
        a_param = AnnotationParam()
        a_param.title = 'Peak Value'
        self.peak_item = AnnotatedPoint(annotationparam=a_param)
        self.plot.add_item(self.peak_item)
        self.plot.set_antialiasing(True)
        self.plot.setCanvasBackground(QtCore.Qt.darkBlue)
        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addWidget(self.plot)
        self.setLayout(vlayout)
        self.update_curve()

    def process_data(self, spectrum, rate):
        """
        Process the input spectrum and return maxiamal
        value of the spectrum
        """
        n = len(spectrum)
        if n == 0:
            self.vx = []
            self.vy = []
            self.x = 0
            self.y = 0
            return (0, 0)
        self.vx = np.arange(0, n//2) * rate / float(n)
        self.vy = spectrum[0:n//2]
        # 6db, 500 hz <--- these parameters are bad for current application
        # minimum_peak_distance = int(500.0 / (rate / float(n)))
        ind = detect_peaks(self.vy, mpd=2)  # around 200 Hz step by default
        # print ind, spectrum[ind]
        # find maximal among ...
        if ind.size:
            self.x = ind[0]
            self.y = spectrum[self.x]
            for i in ind:
                if spectrum[i] > self.y:
                    self.x = i
                    self.y = spectrum[self.x]
            # print(self.x * rate / float(n), self.y)
            # cubic maximize
            left = self.x - 1
            right = self.x + 2
            if left >= 0 and right < n//2:
                data = spectrum[left:right+1]
                (max_pos, max_val) = cubic_maximize(data)
                if max_pos is not None:
                    self.x = left + max_pos
                    self.y = max_val
            # convert to Hz
            self.x *= rate / float(n)
        else:
            self.x = None
            self.y = None
        # print self.x, self.y
        self.update_curve()
        return self.x, self.y

    def update_curve(self):
        # ---Update curve
        if self.vx is not None and self.vy is not None:
            self.curve_item.setBaseline(np.min(self.vy))
            self.curve_item.set_data(self.vx, self.vy)
            self.curve_item.setVisible(True)
        else:
            self.curve_item.setVisible(False)
        if self.x is not None and self.y is not None:
            self.peak_item.set_pos(self.x, self.y)
            self.peak_item.setVisible(True)
        else:
            self.peak_item.setVisible(False)
        self.plot.replot()
        # ---


class AudioAnalyzerForm(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(AudioAnalyzerForm, self).__init__(parent)
        self.settings = QtCore.QSettings(QtCore.QSettings.NativeFormat,
                                         QtCore.QSettings.UserScope,
                                         'BZZ Projects', 'Audio_Analyzer')
        # audio data
        self.audio_data = None
        # audio queue and result queue
        self.audio_q = queue.Queue()
        self.result_q = queue.Queue()
        # create audio listener and audio analyzer thr_slidereads
        self.listener = Listener(self.settings, self.audio_q, self)
        self.analyzer = Analyzer(self.settings,
                                 self.audio_q, self.result_q, self)
        # connect status and result update signal
        self.listener.status.connect(self.updateStatus)
        self.analyzer.analyzed.connect(self.updateResult)
        self.timer = QtCore.QTimer(self)
        self.createMainForm()
        # block the operation buttons if audio input device is not available
        self.timer.singleShot(200, self.checkAvailability)

    def createMainForm(self):
        self.setMinimumSize(800, 600)
        self.setWindowTitle("Audio Analyzer (rev %d)" % REV)
        self.setWindowIcon(loadQIcon(u'icons\Audio_Analyzer.ico'))
        # The main widget
        main_widget = QtWidgets.QWidget(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            main_widget.sizePolicy().hasHeightForWidth())
        main_widget.setSizePolicy(sizePolicy)
        # create widgets ...
        # create QwtPlot widget
        self.curve_widget = FrequencyAnalysisWidget(self)
        self.peak_disp = QtWidgets.QLineEdit()
        self.peak_disp.setAlignment(QtCore.Qt.AlignCenter)
        self.peak_disp.setMinimumHeight(80)
        g1 = self.createSettingWidgets()
        g2 = self.createControlWidgets()
        self.par_layout = QtWidgets.QHBoxLayout()
        self.par_layout.addWidget(g1)
        self.par_layout.addWidget(g2)
        mainLayout = QtWidgets.QVBoxLayout()
        mainLayout.addWidget(self.curve_widget)
        mainLayout.addWidget(self.peak_disp)
        mainLayout.addLayout(self.par_layout)
        main_widget.setLayout(mainLayout)
        self.setCentralWidget(main_widget)
        self.createStatusBar()
        # init contents and actions
        self.loadSettings()
        self.prepareConnections()
        self.setupCurveWidget()
        self.setupPeakDisp()

    def createStatusBar(self):
        self.cur_intensity = QtWidgets.QLabel('Mic intensity: --')
        self.cur_status = QtWidgets.QLabel('Recording: --')

        self.statusBar().addPermanentWidget(self.cur_intensity)
        self.statusBar().addPermanentWidget(self.cur_status)

    def setupCurveWidget(self):
        self.curve_widget.setup_widget('Frequency analysis')
        self.manager = plt.PlotManager(self)
        # ---guiqwt plot manager
        self.manager.add_plot(self.curve_widget.plot)
        toolbar = self.addToolBar('tools')
        self.manager.add_toolbar(toolbar, id(toolbar))
        self.manager.register_all_curve_tools()

    def setupPeakDisp(self):
        self.peak_disp.setReadOnly(True)
        self.peak_disp.setStyleSheet('font: bold 64px;'
                                     'color: yellow;'
                                     'background-color: black;'
                                     'selection-color: yellow;'
                                     'selection-background-color: black;'
                                     )

    def createSettingWidgets(self):
        labelRate = QtWidgets.QLabel('Rate (Hz):')
        labelThr = QtWidgets.QLabel('Threshold:')
        labelSilenceLimit = QtWidgets.QLabel('Silence limit (s):')
        labelPrevAudio = QtWidgets.QLabel('Previous audio (s):')
        self.rate = QtWidgets.QComboBox()
        self.rate.setObjectName(u'rate')
        self.rate.addItems([str(x) for x in AUDIO_RATE])
        alignComboBoxText(self.rate, QtCore.Qt.AlignCenter)
        self.thr_edit = QtWidgets.QLineEdit()
        self.thr_edit.setObjectName(u'threshold')
        self.thr_edit.setValidator(QtGui.QIntValidator(100, 65535))
        self.thr_edit.setAlignment(QtCore.Qt.AlignCenter)
        self.thr_slider = QtWidgets.QSlider()
        self.thr_slider.setObjectName(u'threshold')
        self.thr_slider.setRange(100, 65535)
        self.thr_slider.setSingleStep(10)
        self.thr_slider.setTickInterval(4096)
        self.thr_slider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.thr_slider.setOrientation(QtCore.Qt.Horizontal)
        self.thr_slider.setTracking(False)
        self.silence = QtWidgets.QLineEdit()
        self.silence.setObjectName(u'silence_limit')
        self.silence.setValidator(QtGui.QDoubleValidator(0, 10, 2))
        self.silence.setAlignment(QtCore.Qt.AlignCenter)
        self.prev = QtWidgets.QLineEdit()
        self.prev.setObjectName(u'prev_audio')
        self.prev.setValidator(QtGui.QDoubleValidator(0, 10, 2))
        self.prev.setAlignment(QtCore.Qt.AlignCenter)
        labelWFnc = QtWidgets.QLabel('Window function:')
        labelWSize = QtWidgets.QLabel('Window size:')
        self.w_func = QtWidgets.QComboBox()
        self.w_func.setObjectName(u'window_func')
        self.w_func.addItems(WFUNC_NAMES)
        alignComboBoxText(self.w_func, QtCore.Qt.AlignCenter)
        self.w_size = QtWidgets.QComboBox()
        self.w_size.setObjectName(u'window_size')
        self.w_size.addItems([str(x) for x in WINDOWS_SIZE])
        alignComboBoxText(self.w_size, QtCore.Qt.AlignCenter)
        self.use_freq = QtWidgets.QCheckBox('Display frequency (Hz)')
        self.use_freq.setObjectName(u'use_freq')
        self.use_freq.setToolTip('Uncheck to use G# value')
        layout = QtWidgets.QGridLayout()
        layout.addWidget(labelRate, 0, 0)
        layout.addWidget(self.rate, 0, 1)
        layout.addWidget(labelThr, 1, 0)
        layout.addWidget(self.thr_edit, 1, 1)
        layout.addWidget(self.thr_slider, 2, 0, 1, 2)
        layout.addWidget(labelSilenceLimit, 3, 0)
        layout.addWidget(self.silence, 3, 1)
        layout.addWidget(labelPrevAudio, 4, 0)
        layout.addWidget(self.prev, 4, 1)
        layout.addWidget(labelWFnc, 0, 2, 1, 2)
        layout.addWidget(self.w_func, 1, 2, 1, 2)
        layout.addWidget(labelWSize, 2, 2, 1, 2)
        layout.addWidget(self.w_size, 3, 2, 1, 2)
        layout.addWidget(self.use_freq, 4, 2, 1, 2)
        group = QtWidgets.QGroupBox('Settings')
        group.setLayout(layout)
        return group

    def loadSettings(self):
        n = self.settings.value(self.rate.objectName(),
                                defaultValue=6, type=int)
        self.rate.setCurrentIndex(n)
        n = self.settings.value(self.thr_slider.objectName(),
                                defaultValue=2500, type=int)
        self.thr_slider.setValue(n)
        self.thr_edit.setText(self.settings.value(self.thr_edit.objectName(),
                                                  defaultValue=2500,
                                                  type='QString'))
        self.silence.setText(self.settings.value(self.silence.objectName(),
                                                 defaultValue=1.0,
                                                 type='QString'))
        self.prev.setText(self.settings.value(self.prev.objectName(),
                                              defaultValue=0.5,
                                              type='QString'))
        n = self.settings.value(self.w_func.objectName(),
                                defaultValue=5, type=int)
        self.w_func.setCurrentIndex(n)
        n = self.settings.value(self.w_size.objectName(),
                                defaultValue=4, type=int)
        self.w_size.setCurrentIndex(n)
        n = self.settings.value(self.use_freq.objectName(),
                                defaultValue=True, type=bool)
        self.use_freq.setChecked(n)

    def prepareConnections(self):
        # link threshold slider and threshold edit
        self.thr_slider.valueChanged[int].connect(
            self.onThresholdSliderValueChanged)
        self.thr_edit.textEdited[str].connect(self.onThresholdEditTextEdited)
        # update settings
        self.rate.currentIndexChanged[str].connect(self.updateSettings)
        self.thr_slider.valueChanged[int].connect(self.updateSettings)
        self.thr_edit.textEdited[str].connect(self.updateSettings)
        self.silence.textEdited[str].connect(self.updateSettings)
        self.prev.textEdited[str].connect(self.updateSettings)
        self.w_func.currentIndexChanged[str].connect(self.updateSettings)
        self.w_size.currentIndexChanged[str].connect(self.updateSettings)
        self.use_freq.stateChanged[int].connect(self.updateSettings)
        # actions
        self.btnStart.clicked.connect(self.startAnalysis)
        self.btnStop.clicked.connect(self.stopAnalysis)
        self.btnCheck.clicked.connect(self.checkNoiseLevel)
        self.btnAnalyze.clicked.connect(self.reanalyze)

    def onThresholdSliderValueChanged(self, val):
        try:
            self.thr_edit.setText(str(val))
        except:
            pass

    def onThresholdEditTextEdited(self, val):
        try:
            self.thr_slider.setValue(int(val))
        except:
            pass

    def lockSomeSettingWidgets(self, lock=True):
        """
        some settings should not be updated when analysis has been started
        """
        self.rate.setDisabled(lock)
        self.thr_slider.setDisabled(lock)
        self.thr_edit.setReadOnly(lock)
        self.silence.setReadOnly(lock)
        self.prev.setReadOnly(lock)
        self.btnCheck.setDisabled(lock)

    def createControlWidgets(self):
        self.btnStart = QtWidgets.QPushButton('Start')
        self.btnStop = QtWidgets.QPushButton('Stop')
        self.btnStop.setEnabled(False)
        self.btnCheck = QtWidgets.QPushButton('Noise level')
        self.btnAnalyze = QtWidgets.QPushButton('Re-plot')
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.btnStart, 0, 0, 1, 1)
        layout.addWidget(self.btnStop, 0, 1, 1, 1)
        layout.addWidget(self.btnCheck, 1, 0, 1, 1)
        layout.addWidget(self.btnAnalyze, 1, 1, 1, 1)
        group = QtWidgets.QGroupBox('Controllers')
        group.setLayout(layout)
        return group

    def updateSettings(self, q_str):
        w = self.sender()
        if isinstance(w, QtWidgets.QCheckBox):
            if w.checkState() == QtCore.Qt.Checked:
                self.settings.setValue(w.objectName(), '1')
            else:
                self.settings.setValue(w.objectName(), '0')
        elif isinstance(w, QtWidgets.QLineEdit):
            self.settings.setValue(w.objectName(), w.text())
        elif isinstance(w, QtWidgets.QComboBox):
            self.settings.setValue(w.objectName(),
                                   '{}'.format(w.currentIndex()))
        elif isinstance(w, QtWidgets.QSlider):
            self.statusBar().showMessage(
                'threshold value is set to: {0}'.format(w.value()), 2000)
            self.settings.setValue(w.objectName(),
                                   '{}'.format(w.value()))

    def startAnalysis(self):
        self.listener.start()
        if not self.analyzer.isRunning():
            self.analyzer.start()
        self.lockSomeSettingWidgets(True)
        self.btnStart.setEnabled(False)
        self.btnStop.setEnabled(True)
        self.cur_status.setText('Recording: Waiting')

    def stopAnalysis(self):
        self.listener.quit()
        if self.analyzer.isRunning():
            self.analyzer.quit()
            self.analyzer.wait()
        self.listener.wait()
        self.lockSomeSettingWidgets(False)
        self.btnStop.setEnabled(False)
        self.btnStart.setEnabled(True)
        self.cur_status.setText('Recording: Stopped')

    def updateStatus(self, what, val):
        if what == 'intensity':
            self.cur_intensity.setText('Mic intensity: {}'.format(val))
        elif what == 'status':
            if val == 0:
                msg = 'Recording: Waiting'
            elif val == 1:
                msg = 'Recording: Started'
            elif val == 2:
                msg = 'Recording: Canceled'
            else:
                msg = 'Recording: Stopped'
            self.cur_status.setText(msg)
        else:
            print('UNKOWN message type')

    def updateResult(self):
        self.audio_data, spectrum = self.result_q.get_nowait()
        if spectrum is None:
            # QtGui.QMessageBox.warning(self, 'Audio Analyzer',
            #                          'Recorded audio data is too short.'\
            #                          'Operation abort!')
            return
        rate = self.settings.value('rate', defaultValue=6, type=int)
        hz, db = self.curve_widget.process_data(spectrum, AUDIO_RATE[rate])
        use_freq = self.settings.value('use_freq',
                                       defaultValue=True, type=bool)
        if hz is None or db is None:
            self.statusBar().showMessage('Peak value cannot be located!')
            self.peak_disp.setText(u'00000')
        else:
            self.statusBar().showMessage(
                'Peak Value: {1:.1f} dB @ {0:.1f} Hz'.format(hz, db))
            if use_freq:
                self.peak_disp.setText(
                    u'{1:.1f} dB @ {0:.1f} Hz'.format(hz, db))
            else:
                self.peak_disp.setText(
                    u'Peak G# {:.2f}'.format(2000000.0/hz))

    def checkAvailability(self):
        p = pyaudio.PyAudio()
        try:
            stream = None
            stream = p.open(format=pyaudio.paInt16,
                            channels=1, rate=44100,
                            input=True, frames_per_buffer=1024)
        except IOError:
            self.lockSomeSettingWidgets(True)
            self.btnStart.setEnabled(False)
            self.btnStop.setEnabled(False)
            self.btnCheck.setEnabled(False)
            self.btnAnalyze.setEnabled(False)
            QtWidgets.QMessageBox.critical(
                self, 'Audio Analyzer',
                'No Microphone Attached To the Computer!')
        finally:
            if stream:
                stream.close()
            p.terminate()

    def checkNoiseLevel(self):
        rate = self.settings.value('rate', defaultValue=6, type=int)

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1, rate=AUDIO_RATE[rate],
                        input=True, frames_per_buffer=1024)
        r = get_noise_level(stream, chunk=1024)
        stream.close()
        p.terminate()

        msg = 'Current noise level is {0:.2f}'.format(r)
        self.statusBar().showMessage(msg)
        QtWidgets.QMessageBox.information(self, 'Audio Analyzer', msg)

    def reanalyze(self):
        if self.audio_data is None:
            return
        wfunc = self.settings.value('window_func', defaultValue=5, type=int)
        wsize = self.settings.value('window_size', defaultValue=4, type=int)
        ps = calc_powerspectrum(self.audio_data,
                                WFUNC_TYPES[wfunc], wsize=WINDOWS_SIZE[wsize])
        rate = self.settings.value('rate', defaultValue=6, type=int)
        hz, db = self.curve_widget.process_data(ps, AUDIO_RATE[rate])
        use_freq = self.settings.value('use_freq',
                                       defaultValue=True, type=bool)
        if hz is None or db is None:
            self.statusBar().showMessage('Peak value cannot be located!')
            self.peak_disp.setText(u'00000')
        else:
            self.statusBar().showMessage(
                'Peak Value: {1:.1f} dB @ {0:.1f} Hz'.format(hz, db))
            if use_freq:
                self.peak_disp.setText(
                    u'{1:.1f} dB @ {0:.1f} Hz'.format(hz, db))
            else:
                self.peak_disp.setText(
                    u'Peak G# {:.2f}'.format(2000000.0/hz))

    def closeEvent(self, event):
        self.stopAnalysis()
        self.settings.sync()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # enable anti-alias
    if sys.platform == "win32":
        font = QtGui.QFont("Segoe UI", 9)
    font.setStyleStrategy(QtGui.QFont.PreferAntialias)
    app.setFont(font)
    form = AudioAnalyzerForm()
    form.show()
    sys.exit(app.exec_())
