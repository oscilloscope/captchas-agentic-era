#!/usr/bin/env python3
# main.py
import os
import time
import argparse
from logger import logger
from config import config
from captcha_fsm import solve_captcha_with_state_machine

def main(verbose=False, backend='yolo'):

    logger.set_verbose(verbose)

    logger.always_log(f"Agent ready. Using {backend.upper()} backend. Monitoring for CAPTCHAs.")

    # Initialize the unified processor with the specified backend
    from unified_captcha_processor import create_captcha_processor
    captcha_processor_instance = create_captcha_processor(backend=backend)

    # Set the processor instance in the bridge so FSM uses the correct backend
    from captcha_solver_bridge import set_processor_instance
    set_processor_instance(captcha_processor_instance)
    
    # Create persistent state machine instance to preserve state across retries
    state_machine_instance = None

    try:
        # This is the main monitoring loop. It will run forever.
        while True:
            # Check if checkbox is visible (class_id=4 for 'robot_checkbox')
            checkbox_is_visible = captcha_processor_instance.find_object_on_screen(class_id=4)

            # IMPORTANT: Only check for CAPTCHA dialog if checkbox is not visible
            # The dialog only appears after clicking the checkbox, so no point checking before
            captcha_dialog_visible = None
            if not checkbox_is_visible:
                # Check if CAPTCHA dialog is still open from a previous attempt (class_id=0)
                captcha_dialog_visible = captcha_processor_instance.find_object_on_screen(class_id=0)

            # Show detection status
            if checkbox_is_visible or captcha_dialog_visible:
                status_msg = f"Detection: checkbox={checkbox_is_visible}, dialog={captcha_dialog_visible}"
                logger.always_log(status_msg)

            # If the checkbox is found, start the solver (FSM will handle EVERYTHING)
            if checkbox_is_visible:
                logger.always_log("\nCheckbox detected. Initiating CAPTCHA solver.")

                # Run the state machine to handle the entire workflow.
                # Create new state machine for new CAPTCHA
                success, state_machine_instance = solve_captcha_with_state_machine()

                if success:
                    logger.always_log("\nCAPTCHA solved successfully. Resuming monitoring.")
                    # Clear state machine on success (fresh start for next CAPTCHA)
                    state_machine_instance = None
                else:
                    logger.always_log("\nCAPTCHA solving unsuccessful. Resuming monitoring.")
                    # Keep state_machine_instance for potential retry (preserves previous_cell_data)
                time.sleep(10) 

            # If CAPTCHA dialog is still open but no checkbox (failed state), try to solve it again
            elif captcha_dialog_visible and not checkbox_is_visible:
                logger.always_log("\nCAPTCHA dialog still open. Retrying solution.")

                # Run the state machine again to continue solving
                # CRITICAL: Reuse existing state machine instance to preserve previous_cell_data
                success, state_machine_instance = solve_captcha_with_state_machine(
                    existing_state_machine=state_machine_instance
                )

                if success:
                    logger.always_log("\nRetry successful. CAPTCHA solved.")
                    # Clear state machine on success
                    state_machine_instance = None
                else:
                    logger.always_log("\nRetry unsuccessful. Continuing to monitor.")
                    # Try pressing ESC to close the dialog
                    import pyautogui
                    pyautogui.press('esc')
                    time.sleep(1)
                    # Clear state machine after pressing ESC (starting fresh)
                    state_machine_instance = None
            
            # Wait 2 seconds between each check
            time.sleep(2)

    except KeyboardInterrupt:
        logger.always_log("\nMonitoring stopped by user request.")
    except Exception as e:
        logger.always_log(f"\nAn unexpected error occurred: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated CAPTCHA Solver')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--no-mouse', action='store_true', help='Disable mouse movement animations (instant clicks')
    parser.add_argument('--backend', choices=['yolo', 'llm', 'hybrid'], default='yolo',
                        help='Choose backend: yolo (fast), llm (YOLO detection + LLM solving) or hybrid (best backend per object)')
    args = parser.parse_args()

    # Set mouse movement configuration
    config.set_mouse_movement(not args.no_mouse)
    if args.no_mouse:
        logger.always_log("Mouse movement animations disabled")

    main(verbose=args.verbose, backend=args.backend)