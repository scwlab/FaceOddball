#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on March 18, 2025, at 14:57
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_2
from pylsl import StreamInfo, StreamOutlet
from psychopy import prefs, sound, core
info = StreamInfo(name='PsychoPy',
    type='Markers',
    channel_count=1,
    channel_format='string',
    source_id='PsychopyMarkersID321654'
)
outlet = StreamOutlet(info)

prefs.general['audioLib'] = ['sounddevice']
# Run 'Before Experiment' code from code
import pyaudio
import numpy as np
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'Visual_audio_EP_Oddball'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'tutorial_session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [1680, 1050]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s' % (expInfo['tutorial_session'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='E:\\Projects\\TUT01_VEP_AEP_EEG\\DoubleOddball\\para\\Visual_audio_EP_Oddball.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_block_init') is None:
        # initialise key_resp_block_init
        key_resp_block_init = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_block_init',
        )
    if deviceManager.getDevice('key_resp_block_end') is None:
        # initialise key_resp_block_end
        key_resp_block_end = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_block_end',
        )
    if deviceManager.getDevice('key_resp_block_init_2') is None:
        # initialise key_resp_block_init_2
        key_resp_block_init_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_block_init_2',
        )
    if deviceManager.getDevice('key_resp_block_end_2') is None:
        # initialise key_resp_block_end_2
        key_resp_block_end_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_block_end_2',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "intro" ---
    text_intro = visual.TextStim(win=win, name='text_intro',
        text='Welcome!',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "visual_init" ---
    text_block_init = visual.TextStim(win=win, name='text_block_init',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_block_init = keyboard.Keyboard(deviceName='key_resp_block_init')
    
    # --- Initialize components for Routine "fix" ---
    
    # --- Initialize components for Routine "trial" ---
    
    # --- Initialize components for Routine "block_end" ---
    text_block_end = visual.TextStim(win=win, name='text_block_end',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_block_end = keyboard.Keyboard(deviceName='key_resp_block_end')
    
    # --- Initialize components for Routine "break_btw_oddballs" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='star7',
        size=(0.5, 0.5),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "audio_init" ---
    key_resp_block_init_2 = keyboard.Keyboard(deviceName='key_resp_block_init_2')
    text_block_init_2 = visual.TextStim(win=win, name='text_block_init_2',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "fix" ---
    
    # --- Initialize components for Routine "play_tone" ---
    polygon_2 = visual.ShapeStim(
        win=win, name='polygon_2', vertices='cross',
        size=(0.5, 0.5),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "audio_end" ---
    text_block_end_2 = visual.TextStim(win=win, name='text_block_end_2',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_block_end_2 = keyboard.Keyboard(deviceName='key_resp_block_end_2')
    
    # --- Initialize components for Routine "end" ---
    text_end = visual.TextStim(win=win, name='text_end',
        text='This is the end of the experiment.\n\nThank you for participating!',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "intro" ---
    # create an object to store info about Routine intro
    intro = data.Routine(
        name='intro',
        components=[text_intro],
    )
    intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for intro
    intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro.tStart = globalClock.getTime(format='float')
    intro.status = STARTED
    thisExp.addData('intro.started', intro.tStart)
    intro.maxDuration = None
    # keep track of which components have finished
    introComponents = intro.components
    for thisComponent in intro.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro" ---
    intro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_intro* updates
        
        # if text_intro is starting this frame...
        if text_intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro.frameNStart = frameN  # exact frame index
            text_intro.tStart = t  # local t and not account for scr refresh
            text_intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_intro.started')
            # update status
            text_intro.status = STARTED
            text_intro.setAutoDraw(True)
        
        # if text_intro is active this frame...
        if text_intro.status == STARTED:
            # update params
            pass
        
        # if text_intro is stopping this frame...
        if text_intro.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_intro.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                text_intro.tStop = t  # not accounting for scr refresh
                text_intro.tStopRefresh = tThisFlipGlobal  # on global time
                text_intro.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_intro.stopped')
                # update status
                text_intro.status = FINISHED
                text_intro.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            intro.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro" ---
    for thisComponent in intro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro
    intro.tStop = globalClock.getTime(format='float')
    intro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro.stopped', intro.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if intro.maxDurationReached:
        routineTimer.addTime(-intro.maxDuration)
    elif intro.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    blocks_visual = data.TrialHandler2(
        name='blocks_visual',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('lists/blocks.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(blocks_visual)  # add the loop to the experiment
    thisBlocks_visual = blocks_visual.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlocks_visual.rgb)
    if thisBlocks_visual != None:
        for paramName in thisBlocks_visual:
            globals()[paramName] = thisBlocks_visual[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlocks_visual in blocks_visual:
        currentLoop = blocks_visual
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlocks_visual.rgb)
        if thisBlocks_visual != None:
            for paramName in thisBlocks_visual:
                globals()[paramName] = thisBlocks_visual[paramName]
        
        # --- Prepare to start Routine "visual_init" ---
        # create an object to store info about Routine visual_init
        visual_init = data.Routine(
            name='visual_init',
            components=[text_block_init, key_resp_block_init],
        )
        visual_init.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_block_init.setText(f"""
        This is block number {block}
        
        [Press space to continue]
        """)
        # create starting attributes for key_resp_block_init
        key_resp_block_init.keys = []
        key_resp_block_init.rt = []
        _key_resp_block_init_allKeys = []
        # store start times for visual_init
        visual_init.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        visual_init.tStart = globalClock.getTime(format='float')
        visual_init.status = STARTED
        thisExp.addData('visual_init.started', visual_init.tStart)
        visual_init.maxDuration = None
        # keep track of which components have finished
        visual_initComponents = visual_init.components
        for thisComponent in visual_init.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "visual_init" ---
        # if trial has changed, end Routine now
        if isinstance(blocks_visual, data.TrialHandler2) and thisBlocks_visual.thisN != blocks_visual.thisTrial.thisN:
            continueRoutine = False
        visual_init.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_block_init* updates
            
            # if text_block_init is starting this frame...
            if text_block_init.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_block_init.frameNStart = frameN  # exact frame index
                text_block_init.tStart = t  # local t and not account for scr refresh
                text_block_init.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_block_init, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_block_init.started')
                # update status
                text_block_init.status = STARTED
                text_block_init.setAutoDraw(True)
            
            # if text_block_init is active this frame...
            if text_block_init.status == STARTED:
                # update params
                pass
            
            # *key_resp_block_init* updates
            waitOnFlip = False
            
            # if key_resp_block_init is starting this frame...
            if key_resp_block_init.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_block_init.frameNStart = frameN  # exact frame index
                key_resp_block_init.tStart = t  # local t and not account for scr refresh
                key_resp_block_init.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_block_init, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_block_init.started')
                # update status
                key_resp_block_init.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_block_init.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_block_init.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_block_init.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_block_init.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_block_init_allKeys.extend(theseKeys)
                if len(_key_resp_block_init_allKeys):
                    key_resp_block_init.keys = _key_resp_block_init_allKeys[-1].name  # just the last key pressed
                    key_resp_block_init.rt = _key_resp_block_init_allKeys[-1].rt
                    key_resp_block_init.duration = _key_resp_block_init_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                visual_init.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in visual_init.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "visual_init" ---
        for thisComponent in visual_init.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for visual_init
        visual_init.tStop = globalClock.getTime(format='float')
        visual_init.tStopRefresh = tThisFlipGlobal
        thisExp.addData('visual_init.stopped', visual_init.tStop)
        # check responses
        if key_resp_block_init.keys in ['', [], None]:  # No response was made
            key_resp_block_init.keys = None
        blocks_visual.addData('key_resp_block_init.keys',key_resp_block_init.keys)
        if key_resp_block_init.keys != None:  # we had a response
            blocks_visual.addData('key_resp_block_init.rt', key_resp_block_init.rt)
            blocks_visual.addData('key_resp_block_init.duration', key_resp_block_init.duration)
        # the Routine "visual_init" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "fix" ---
        # create an object to store info about Routine fix
        fix = data.Routine(
            name='fix',
            components=[],
        )
        fix.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fix
        fix.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fix.tStart = globalClock.getTime(format='float')
        fix.status = STARTED
        thisExp.addData('fix.started', fix.tStart)
        fix.maxDuration = None
        # keep track of which components have finished
        fixComponents = fix.components
        for thisComponent in fix.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fix" ---
        # if trial has changed, end Routine now
        if isinstance(blocks_visual, data.TrialHandler2) and thisBlocks_visual.thisN != blocks_visual.thisTrial.thisN:
            continueRoutine = False
        fix.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fix.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fix.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fix" ---
        for thisComponent in fix.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fix
        fix.tStop = globalClock.getTime(format='float')
        fix.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fix.stopped', fix.tStop)
        # the Routine "fix" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('lists/trials_faces.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if trials.trialList in ([], [None], None):
            params = []
        else:
            params = trials.trialList[0].keys()
        # save data for this loop
        trials.saveAsText(filename + 'trials.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "block_end" ---
        # create an object to store info about Routine block_end
        block_end = data.Routine(
            name='block_end',
            components=[text_block_end, key_resp_block_end],
        )
        block_end.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_block_end.setText(f"""
        This is the end of block number {block}
        
        [Press space to continue]
        """)
        # create starting attributes for key_resp_block_end
        key_resp_block_end.keys = []
        key_resp_block_end.rt = []
        _key_resp_block_end_allKeys = []
        # store start times for block_end
        block_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block_end.tStart = globalClock.getTime(format='float')
        block_end.status = STARTED
        thisExp.addData('block_end.started', block_end.tStart)
        block_end.maxDuration = None
        # keep track of which components have finished
        block_endComponents = block_end.components
        for thisComponent in block_end.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "block_end" ---
        # if trial has changed, end Routine now
        if isinstance(blocks_visual, data.TrialHandler2) and thisBlocks_visual.thisN != blocks_visual.thisTrial.thisN:
            continueRoutine = False
        block_end.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_block_end* updates
            
            # if text_block_end is starting this frame...
            if text_block_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_block_end.frameNStart = frameN  # exact frame index
                text_block_end.tStart = t  # local t and not account for scr refresh
                text_block_end.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_block_end, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_block_end.started')
                # update status
                text_block_end.status = STARTED
                text_block_end.setAutoDraw(True)
            
            # if text_block_end is active this frame...
            if text_block_end.status == STARTED:
                # update params
                pass
            
            # *key_resp_block_end* updates
            waitOnFlip = False
            
            # if key_resp_block_end is starting this frame...
            if key_resp_block_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_block_end.frameNStart = frameN  # exact frame index
                key_resp_block_end.tStart = t  # local t and not account for scr refresh
                key_resp_block_end.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_block_end, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_block_end.started')
                # update status
                key_resp_block_end.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_block_end.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_block_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_block_end.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_block_end.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_block_end_allKeys.extend(theseKeys)
                if len(_key_resp_block_end_allKeys):
                    key_resp_block_end.keys = _key_resp_block_end_allKeys[-1].name  # just the last key pressed
                    key_resp_block_end.rt = _key_resp_block_end_allKeys[-1].rt
                    key_resp_block_end.duration = _key_resp_block_end_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                block_end.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block_end.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "block_end" ---
        for thisComponent in block_end.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block_end
        block_end.tStop = globalClock.getTime(format='float')
        block_end.tStopRefresh = tThisFlipGlobal
        thisExp.addData('block_end.stopped', block_end.tStop)
        # check responses
        if key_resp_block_end.keys in ['', [], None]:  # No response was made
            key_resp_block_end.keys = None
        blocks_visual.addData('key_resp_block_end.keys',key_resp_block_end.keys)
        if key_resp_block_end.keys != None:  # we had a response
            blocks_visual.addData('key_resp_block_end.rt', key_resp_block_end.rt)
            blocks_visual.addData('key_resp_block_end.duration', key_resp_block_end.duration)
        # the Routine "block_end" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'blocks_visual'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if blocks_visual.trialList in ([], [None], None):
        params = []
    else:
        params = blocks_visual.trialList[0].keys()
    # save data for this loop
    blocks_visual.saveAsText(filename + 'blocks_visual.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "break_btw_oddballs" ---
    # create an object to store info about Routine break_btw_oddballs
    break_btw_oddballs = data.Routine(
        name='break_btw_oddballs',
        components=[polygon],
    )
    break_btw_oddballs.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_btw_oddballs
    break_btw_oddballs.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_btw_oddballs.tStart = globalClock.getTime(format='float')
    break_btw_oddballs.status = STARTED
    thisExp.addData('break_btw_oddballs.started', break_btw_oddballs.tStart)
    break_btw_oddballs.maxDuration = None
    # keep track of which components have finished
    break_btw_oddballsComponents = break_btw_oddballs.components
    for thisComponent in break_btw_oddballs.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_btw_oddballs" ---
    break_btw_oddballs.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *polygon* updates
        
        # if polygon is starting this frame...
        if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon.frameNStart = frameN  # exact frame index
            polygon.tStart = t  # local t and not account for scr refresh
            polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon.started')
            # update status
            polygon.status = STARTED
            polygon.setAutoDraw(True)
        
        # if polygon is active this frame...
        if polygon.status == STARTED:
            # update params
            pass
        
        # if polygon is stopping this frame...
        if polygon.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > polygon.tStartRefresh + 1-frameTolerance:
                # keep track of stop time/frame for later
                polygon.tStop = t  # not accounting for scr refresh
                polygon.tStopRefresh = tThisFlipGlobal  # on global time
                polygon.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon.stopped')
                # update status
                polygon.status = FINISHED
                polygon.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_btw_oddballs.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_btw_oddballs.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_btw_oddballs" ---
    for thisComponent in break_btw_oddballs.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_btw_oddballs
    break_btw_oddballs.tStop = globalClock.getTime(format='float')
    break_btw_oddballs.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_btw_oddballs.stopped', break_btw_oddballs.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_btw_oddballs.maxDurationReached:
        routineTimer.addTime(-break_btw_oddballs.maxDuration)
    elif break_btw_oddballs.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    blocks_audio = data.TrialHandler2(
        name='blocks_audio',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('lists/blocks.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(blocks_audio)  # add the loop to the experiment
    thisBlocks_audio = blocks_audio.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlocks_audio.rgb)
    if thisBlocks_audio != None:
        for paramName in thisBlocks_audio:
            globals()[paramName] = thisBlocks_audio[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlocks_audio in blocks_audio:
        currentLoop = blocks_audio
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlocks_audio.rgb)
        if thisBlocks_audio != None:
            for paramName in thisBlocks_audio:
                globals()[paramName] = thisBlocks_audio[paramName]
        
        # --- Prepare to start Routine "audio_init" ---
        # create an object to store info about Routine audio_init
        audio_init = data.Routine(
            name='audio_init',
            components=[key_resp_block_init_2, text_block_init_2],
        )
        audio_init.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp_block_init_2
        key_resp_block_init_2.keys = []
        key_resp_block_init_2.rt = []
        _key_resp_block_init_2_allKeys = []
        text_block_init_2.setText(f"""
        This is block number {block}
        
        [Press space to continue]
        """)
        # store start times for audio_init
        audio_init.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        audio_init.tStart = globalClock.getTime(format='float')
        audio_init.status = STARTED
        thisExp.addData('audio_init.started', audio_init.tStart)
        audio_init.maxDuration = None
        # keep track of which components have finished
        audio_initComponents = audio_init.components
        for thisComponent in audio_init.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "audio_init" ---
        # if trial has changed, end Routine now
        if isinstance(blocks_audio, data.TrialHandler2) and thisBlocks_audio.thisN != blocks_audio.thisTrial.thisN:
            continueRoutine = False
        audio_init.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_resp_block_init_2* updates
            waitOnFlip = False
            
            # if key_resp_block_init_2 is starting this frame...
            if key_resp_block_init_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_block_init_2.frameNStart = frameN  # exact frame index
                key_resp_block_init_2.tStart = t  # local t and not account for scr refresh
                key_resp_block_init_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_block_init_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_block_init_2.started')
                # update status
                key_resp_block_init_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_block_init_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_block_init_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_block_init_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_block_init_2.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_block_init_2_allKeys.extend(theseKeys)
                if len(_key_resp_block_init_2_allKeys):
                    key_resp_block_init_2.keys = _key_resp_block_init_2_allKeys[-1].name  # just the last key pressed
                    key_resp_block_init_2.rt = _key_resp_block_init_2_allKeys[-1].rt
                    key_resp_block_init_2.duration = _key_resp_block_init_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_block_init_2* updates
            
            # if text_block_init_2 is starting this frame...
            if text_block_init_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_block_init_2.frameNStart = frameN  # exact frame index
                text_block_init_2.tStart = t  # local t and not account for scr refresh
                text_block_init_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_block_init_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_block_init_2.started')
                # update status
                text_block_init_2.status = STARTED
                text_block_init_2.setAutoDraw(True)
            
            # if text_block_init_2 is active this frame...
            if text_block_init_2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                audio_init.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in audio_init.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "audio_init" ---
        for thisComponent in audio_init.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for audio_init
        audio_init.tStop = globalClock.getTime(format='float')
        audio_init.tStopRefresh = tThisFlipGlobal
        thisExp.addData('audio_init.stopped', audio_init.tStop)
        # check responses
        if key_resp_block_init_2.keys in ['', [], None]:  # No response was made
            key_resp_block_init_2.keys = None
        blocks_audio.addData('key_resp_block_init_2.keys',key_resp_block_init_2.keys)
        if key_resp_block_init_2.keys != None:  # we had a response
            blocks_audio.addData('key_resp_block_init_2.rt', key_resp_block_init_2.rt)
            blocks_audio.addData('key_resp_block_init_2.duration', key_resp_block_init_2.duration)
        # the Routine "audio_init" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "fix" ---
        # create an object to store info about Routine fix
        fix = data.Routine(
            name='fix',
            components=[],
        )
        fix.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fix
        fix.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fix.tStart = globalClock.getTime(format='float')
        fix.status = STARTED
        thisExp.addData('fix.started', fix.tStart)
        fix.maxDuration = None
        # keep track of which components have finished
        fixComponents = fix.components
        for thisComponent in fix.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fix" ---
        # if trial has changed, end Routine now
        if isinstance(blocks_audio, data.TrialHandler2) and thisBlocks_audio.thisN != blocks_audio.thisTrial.thisN:
            continueRoutine = False
        fix.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fix.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fix.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fix" ---
        for thisComponent in fix.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fix
        fix.tStop = globalClock.getTime(format='float')
        fix.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fix.stopped', fix.tStop)
        # the Routine "fix" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_2 = data.TrialHandler2(
            name='trials_2',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('lists/trials_audio.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(trials_2)  # add the loop to the experiment
        thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial_2 in trials_2:
            currentLoop = trials_2
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
            if thisTrial_2 != None:
                for paramName in thisTrial_2:
                    globals()[paramName] = thisTrial_2[paramName]
            
            # --- Prepare to start Routine "play_tone" ---
            # create an object to store info about Routine play_tone
            play_tone = data.Routine(
                name='play_tone',
                components=[polygon_2],
            )
            play_tone.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code
            outlet.push_sample([category])
                                                            
            p = pyaudio.Audio()
            volume = 0.5
            fs = 44100
            duration = 0.5
            f = 440
            
            samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)
            output_bytes = (volume * samples).tobytes()
            stream = p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=fs,
                            output=True)
            stream.write(output_bytes)
            stream.stop_stream()
            stream.close()
            p.terminate()
            # store start times for play_tone
            play_tone.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            play_tone.tStart = globalClock.getTime(format='float')
            play_tone.status = STARTED
            thisExp.addData('play_tone.started', play_tone.tStart)
            play_tone.maxDuration = None
            # keep track of which components have finished
            play_toneComponents = play_tone.components
            for thisComponent in play_tone.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "play_tone" ---
            # if trial has changed, end Routine now
            if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
                continueRoutine = False
            play_tone.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *polygon_2* updates
                
                # if polygon_2 is starting this frame...
                if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    polygon_2.frameNStart = frameN  # exact frame index
                    polygon_2.tStart = t  # local t and not account for scr refresh
                    polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_2.started')
                    # update status
                    polygon_2.status = STARTED
                    polygon_2.setAutoDraw(True)
                
                # if polygon_2 is active this frame...
                if polygon_2.status == STARTED:
                    # update params
                    pass
                
                # if polygon_2 is stopping this frame...
                if polygon_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > polygon_2.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        polygon_2.tStop = t  # not accounting for scr refresh
                        polygon_2.tStopRefresh = tThisFlipGlobal  # on global time
                        polygon_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'polygon_2.stopped')
                        # update status
                        polygon_2.status = FINISHED
                        polygon_2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    play_tone.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in play_tone.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "play_tone" ---
            for thisComponent in play_tone.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for play_tone
            play_tone.tStop = globalClock.getTime(format='float')
            play_tone.tStopRefresh = tThisFlipGlobal
            thisExp.addData('play_tone.stopped', play_tone.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if play_tone.maxDurationReached:
                routineTimer.addTime(-play_tone.maxDuration)
            elif play_tone.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials_2'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if trials_2.trialList in ([], [None], None):
            params = []
        else:
            params = trials_2.trialList[0].keys()
        # save data for this loop
        trials_2.saveAsText(filename + 'trials_2.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "audio_end" ---
        # create an object to store info about Routine audio_end
        audio_end = data.Routine(
            name='audio_end',
            components=[text_block_end_2, key_resp_block_end_2],
        )
        audio_end.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_block_end_2.setText(f"""
        This is the end of block number {block}
        
        [Press space to continue]
        """)
        # create starting attributes for key_resp_block_end_2
        key_resp_block_end_2.keys = []
        key_resp_block_end_2.rt = []
        _key_resp_block_end_2_allKeys = []
        # store start times for audio_end
        audio_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        audio_end.tStart = globalClock.getTime(format='float')
        audio_end.status = STARTED
        thisExp.addData('audio_end.started', audio_end.tStart)
        audio_end.maxDuration = None
        # keep track of which components have finished
        audio_endComponents = audio_end.components
        for thisComponent in audio_end.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "audio_end" ---
        # if trial has changed, end Routine now
        if isinstance(blocks_audio, data.TrialHandler2) and thisBlocks_audio.thisN != blocks_audio.thisTrial.thisN:
            continueRoutine = False
        audio_end.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_block_end_2* updates
            
            # if text_block_end_2 is starting this frame...
            if text_block_end_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_block_end_2.frameNStart = frameN  # exact frame index
                text_block_end_2.tStart = t  # local t and not account for scr refresh
                text_block_end_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_block_end_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_block_end_2.started')
                # update status
                text_block_end_2.status = STARTED
                text_block_end_2.setAutoDraw(True)
            
            # if text_block_end_2 is active this frame...
            if text_block_end_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_block_end_2* updates
            waitOnFlip = False
            
            # if key_resp_block_end_2 is starting this frame...
            if key_resp_block_end_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_block_end_2.frameNStart = frameN  # exact frame index
                key_resp_block_end_2.tStart = t  # local t and not account for scr refresh
                key_resp_block_end_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_block_end_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_block_end_2.started')
                # update status
                key_resp_block_end_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_block_end_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_block_end_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_block_end_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_block_end_2.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_block_end_2_allKeys.extend(theseKeys)
                if len(_key_resp_block_end_2_allKeys):
                    key_resp_block_end_2.keys = _key_resp_block_end_2_allKeys[-1].name  # just the last key pressed
                    key_resp_block_end_2.rt = _key_resp_block_end_2_allKeys[-1].rt
                    key_resp_block_end_2.duration = _key_resp_block_end_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                audio_end.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in audio_end.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "audio_end" ---
        for thisComponent in audio_end.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for audio_end
        audio_end.tStop = globalClock.getTime(format='float')
        audio_end.tStopRefresh = tThisFlipGlobal
        thisExp.addData('audio_end.stopped', audio_end.tStop)
        # check responses
        if key_resp_block_end_2.keys in ['', [], None]:  # No response was made
            key_resp_block_end_2.keys = None
        blocks_audio.addData('key_resp_block_end_2.keys',key_resp_block_end_2.keys)
        if key_resp_block_end_2.keys != None:  # we had a response
            blocks_audio.addData('key_resp_block_end_2.rt', key_resp_block_end_2.rt)
            blocks_audio.addData('key_resp_block_end_2.duration', key_resp_block_end_2.duration)
        # the Routine "audio_end" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'blocks_audio'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if blocks_audio.trialList in ([], [None], None):
        params = []
    else:
        params = blocks_audio.trialList[0].keys()
    # save data for this loop
    blocks_audio.saveAsText(filename + 'blocks_audio.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[text_end],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end* updates
        
        # if text_end is starting this frame...
        if text_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end.frameNStart = frameN  # exact frame index
            text_end.tStart = t  # local t and not account for scr refresh
            text_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end.started')
            # update status
            text_end.status = STARTED
            text_end.setAutoDraw(True)
        
        # if text_end is active this frame...
        if text_end.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
