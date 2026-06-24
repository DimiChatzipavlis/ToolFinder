"""Author the unseen-server (regime 3) evaluation queries.

Hand-written by a different model family (Claude) than the original dataset
generator, against tools sampled from the multi-server catalog. Authoring
rules, applied per query: express the user's goal in natural language; avoid
copying operationId tokens or description phrasing where a natural alternative
exists; vary register across imperative / interrogative / conversational forms.
Eval-only: these queries are never used for training, so regime 3 measures
zero-shot transfer of GitHub-trained systems to 23 unseen servers among 544
distractor tools.

Outputs:
    experiments/data/queries_multiserver.csv
    experiments/data/corpus_multiserver.json   (30 GitHub + 544 OpenAPI tools)
    experiments/data/splits/regime3_unseen_servers.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402

QUERIES: dict[str, list[str]] = {
    "asana/getcustomfield": [
        "Show me how that custom field is defined, including its type and settings.",
        "What are the metadata details of the priority custom field?",
        "I need the full definition of the custom field we use for sprint tracking.",
    ],
    "asana/creategoal": [
        "Set up a new quarterly objective for the marketing team workspace.",
        "Can you add a goal called 'Reduce churn by 10%' to our team?",
        "We agreed on a new target last planning meeting - please record it as a goal.",
    ],
    "asana/insertenumoptionforcustomfield": [
        "Move the 'Blocked' option above 'In Review' in that dropdown field.",
        "Can you reorder the choices of our status field so Critical comes first?",
        "The dropdown options are in the wrong order - put Low after Medium.",
    ],
    "atlassian/getallsystemavatars": [
        "List the built-in avatars available for projects.",
        "What default avatar images does Jira offer for issue types?",
        "Show every stock avatar we can assign, grouped by owner type.",
    ],
    "atlassian/getapplicationrole": [
        "Show the details of the jira-software application role.",
        "What does the application role with key jira-core contain?",
        "I want to inspect one specific application access role on our Jira site.",
    ],
    "atlassian/getallapplicationroles": [
        "List every application access role configured on this Jira instance.",
        "Which application roles exist and which groups are assigned to them?",
        "Give me an overview of all product access roles in Jira.",
    ],
    "box/put_comments_id": [
        "Change the text of my comment on that contract file.",
        "Edit the comment I left yesterday so it says 'final version approved'.",
        "Fix the typo in comment 88123 on the shared document.",
    ],
    "box/delete_collaborations_id": [
        "Remove the external partner's access to the shared folder.",
        "Revoke that collaboration we set up for the contractor.",
        "Take away the agency's shared access - the engagement ended.",
    ],
    "box/get_collaboration_whitelist_entries_id": [
        "Show the details of that approved collaboration domain entry.",
        "Which domain does allow-list entry 4321 cover?",
        "Look up the safelisted external domain record by its id.",
    ],
    "circleci/post_project_username_project_build_num_cancel": [
        "Stop build 1542 of the backend project, it's stuck.",
        "Cancel the running CI build for our web app.",
        "That pipeline run is wasting credits - kill it.",
    ],
    "circleci/post_project_username_project_ssh_key": [
        "Add an SSH key so the build can reach our private artifact server.",
        "The deploy step needs key-based access to the bastion - set that up for the project.",
        "Register a new SSH identity for this repository's CI jobs.",
    ],
    "circleci/get_me": [
        "Which account am I logged into CircleCI as?",
        "Show my own CI user profile.",
        "Who am I according to the CI service?",
    ],
    "digitalocean/oneclicks_list": [
        "What one-click applications can I install on a droplet?",
        "List the ready-made app images available in the marketplace.",
        "Show me every 1-click app option, including kubernetes ones.",
    ],
    "digitalocean/apps_list_regions": [
        "Which regions can App Platform deploy to?",
        "Show the datacenter locations supported for app hosting.",
        "Where geographically can I run an app on this platform?",
    ],
    "digitalocean/apps_get_deployment": [
        "Show the status of deployment 7f2a of the storefront app.",
        "Get the details for that specific app deployment, including its phase.",
        "What happened with the latest deployment of my app? Pull its record.",
    ],
    "docusign/billingplan_getdowngraderequestbillinginfo": [
        "What would downgrading our e-signature plan look like billing-wise?",
        "Show the billing details for the pending plan downgrade on this account.",
        "We asked to move to a cheaper plan - retrieve the downgrade information.",
    ],
    "docusign/serviceinformation_getresourceinformation": [
        "List the base endpoints this e-signature API version exposes.",
        "What top-level resources are available in the REST service?",
        "Enumerate the API's root resources for the current version.",
    ],
    "docusign/brands_deletebrands": [
        "Remove the old company branding profiles from the account.",
        "Delete the two outdated signing brands we no longer use.",
        "Clean up the unused brand themes from our e-signature settings.",
    ],
    "launchdarkly/getroot": [
        "What resource categories does the feature flag API expose at its root?",
        "Show the top-level resources I can navigate to in LaunchDarkly.",
        "Hit the API root and tell me which collections exist.",
    ],
    "launchdarkly/getauditlogentries": [
        "Who changed flag settings last week? Pull the audit trail.",
        "Show the audit log entries between Monday and Friday.",
        "I need the change history records filtered by date for compliance.",
    ],
    "launchdarkly/getfeatureflag": [
        "Show the configuration of the dark-mode feature flag.",
        "What's the current state of the flag with key checkout-redesign?",
        "Pull up that one feature toggle so I can check its targeting rules.",
    ],
    "linode/getclientthumbnail": [
        "Fetch the logo image for my OAuth client.",
        "Show the thumbnail associated with that OAuth application.",
        "Get the small icon registered for client 1234.",
    ],
    "linode/resetclientsecret": [
        "The OAuth secret leaked - rotate it for my client now.",
        "Generate a fresh secret for the OAuth app I own.",
        "Invalidate and replace the client secret for our integration.",
    ],
    "linode/eventseen": [
        "Mark all my notifications up to event 999 as read.",
        "Clear the unread state on events older than that one.",
        "Flag everything up to and including this event as seen.",
    ],
    "medium/get_article_article_id_markdown": [
        "Give me the markdown source of that Medium story.",
        "Export the article as markdown so I can republish it.",
        "Fetch the raw markdown for post 3f9c on Medium.",
    ],
    "medium/get_publication_publication_id_newsletter": [
        "Show the newsletter details for the Better Programming publication.",
        "What newsletter is attached to that Medium publication?",
        "Pull the newsletter name, slug and subscriber info for this publication.",
    ],
    "medium/get_top_writer_topic_slug": [
        "Who are the leading authors writing about machine learning on Medium?",
        "List the top writers in the productivity topic.",
        "Find the most influential contributors for the javascript niche.",
    ],
    "netlify/deleteenvvarvalue": [
        "Remove the staging value of the DATABASE_URL environment variable.",
        "Delete that one environment variable value from the site settings.",
        "Drop the deploy-context-specific value of the API_KEY variable.",
    ],
    "netlify/listsites": [
        "Show all the sites in my Netlify account.",
        "Which websites am I currently hosting there?",
        "List my deployed sites with their settings.",
    ],
    "plaid/banktransfersweeplist": [
        "Show the recent sweep transactions matching last month's transfers.",
        "List the sweeps that settled funds to our account in March.",
        "Pull the sweep history with the usual filters.",
    ],
    "plaid/banktransfereventlist": [
        "List the ACH events for the transfers we initiated this week.",
        "What status events were recorded for our bank transfers?",
        "Show me the transfer event stream so I can debug the failed payout.",
    ],
    "plaid/banktransfersweepget": [
        "Get the details of sweep sw_42.",
        "Look up that one sweep record by its identifier.",
        "Retrieve the sweep that corresponds to yesterday's settlement.",
    ],
    "postmarkapp/requestdkimverificationfordomain": [
        "Trigger the DKIM DNS check for example.com.",
        "Ask the mail service to verify our domain's DKIM record now.",
        "We just added the DNS entry - request DKIM verification for the domain.",
    ],
    "postmarkapp/requestreturnpathverificationfordomain": [
        "Verify the return-path DNS record for our sending domain.",
        "Kick off the bounce-domain verification for example.org.",
        "The CNAME for the return path is live - please run the verification.",
    ],
    "postmarkapp/requestnewdkimkeyforsendersignature": [
        "Rotate the DKIM key for this sender signature.",
        "Issue a fresh DKIM key pair for our outbound email identity.",
        "Generate a new DKIM key; we'll update DNS once it's pending.",
    ],
    "sendgrid/get_api_keys_api_key_id": [
        "Show the details of the API key ending in 9Xk2.",
        "What permissions does that one SendGrid key have?",
        "Look up the email API key by its identifier.",
    ],
    "sendgrid/delete_access_settings_whitelist": [
        "Remove the office IPs from the email service allow list.",
        "Delete those two addresses from the IP access list.",
        "Take 203.0.113.7 off the allowed IP ranges for the account.",
    ],
    "sendgrid/get_access_settings_whitelist_rule_id": [
        "Show which IP address allow-list rule 15 refers to.",
        "Get the details of that specific allowed IP entry.",
        "Look up one IP access rule by its id.",
    ],
    "slack/admin_conversations_restrictaccess_removegroup": [
        "Unlink the contractors IDP group from the private finance channel.",
        "Remove that identity-provider group's access restriction from the channel.",
        "Detach the SSO group from #leadership so access is no longer gated.",
    ],
    "slack/admin_apps_approve": [
        "Approve the Figma app for installation in our workspace.",
        "Green-light that pending app request for the engineering workspace.",
        "As an admin, allow the requested integration to be installed.",
    ],
    "slack/admin_conversations_invite": [
        "Add the new hire to the #onboarding channel.",
        "Invite Priya to the private incident-response channel.",
        "Put these three users into #release-coordination.",
    ],
    "spoonacular/detectfoodintext": [
        "Find every food item mentioned in this restaurant review.",
        "Which ingredients or dishes appear in the following paragraph?",
        "Scan my meal diary text and extract the foods it mentions.",
    ],
    "spoonacular/getingredientsubstitutesbyid": [
        "What can I use instead of buttermilk in this recipe?",
        "Suggest replacements for ingredient 9314.",
        "I'm out of that ingredient - what are good substitutes for it?",
    ],
    "spoonacular/menuitemnutritionbyidimage": [
        "Render the nutrition facts image for that menu item.",
        "Show a visual nutrition label for the Big Mac entry.",
        "Generate the calorie and macro graphic for menu item 424571.",
    ],
    "spotify/get_a_chapter": [
        "Get the details of chapter 3 of that audiobook.",
        "Show information about a single audiobook chapter.",
        "Pull the metadata for the chapter with this catalog id.",
    ],
    "spotify/get_available_markets": [
        "In which countries is the streaming service available?",
        "List the market codes the platform supports.",
        "Show all regions where users can access the catalog.",
    ],
    "spotify/get_current_users_profile": [
        "Show my own streaming profile details.",
        "What's my display name and subscription level on the music service?",
        "Fetch the account info of the currently signed-in listener.",
    ],
    "squareup/getbankaccountbyv1id": [
        "Look up the linked bank account using its old v1 identifier.",
        "Fetch the bank account details for that legacy id.",
        "Get the banking record referenced by the v1-era account id.",
    ],
    "squareup/retrieveorder": [
        "Show everything about online order ORD-5521.",
        "Pull the full history and line items of that customer's order.",
        "Get the details for one specific store order.",
    ],
    "squareup/obtaintoken": [
        "Exchange the authorization code for an access token.",
        "Get me a fresh OAuth token using the refresh flow.",
        "Complete the OAuth handshake and return the bearer token.",
    ],
    "stripe/deleteaccountsaccountbankaccountsid": [
        "Remove the old payout bank account from the connected account.",
        "Delete that external bank account from the merchant's profile.",
        "Drop the unused bank account ba_1Nx from account acct_77.",
    ],
    "stripe/getaccountsaccountexternalaccounts": [
        "List the payout destinations attached to this connected account.",
        "Which external bank accounts or cards does the merchant have on file?",
        "Show all external accounts for acct_4Qz.",
    ],
    "stripe/deleteaccountsaccountpersonsperson": [
        "Remove the former director from the account's legal entity.",
        "Delete that person's association with the connected account.",
        "The co-owner left the company - detach their person record.",
    ],
    "twilio/updatecallrecording": [
        "Pause the recording of the ongoing support call.",
        "Stop recording the current call immediately.",
        "Resume the paused recording on that active call.",
    ],
    "twilio/fetchcallrecording": [
        "Get the recording from yesterday's customer call.",
        "Retrieve that one call recording by its id.",
        "Pull the audio recording attached to call CA9981.",
    ],
    "twilio/listcallevent": [
        "Show the event timeline for that phone call.",
        "List everything that happened during call CA1234.",
        "Pull the per-call event log so I can debug the dropped audio.",
    ],
    "twitter/findspacesbycreatorids": [
        "Which audio Spaces have these two accounts hosted?",
        "Find the live rooms created by user 783214.",
        "Look up Spaces by their creators' user ids.",
    ],
    "twitter/listidcreate": [
        "Make a new list called 'AI researchers'.",
        "Create a private list for industry news accounts.",
        "Set up a fresh list so I can group the design community.",
    ],
    "twitter/listsidtweets": [
        "Show the latest posts from my 'AI researchers' list.",
        "What tweets are in the timeline of list 8842?",
        "Fetch the recent tweets belonging to that list.",
    ],
    "vercel/deleteedgeconfigtokens": [
        "Revoke the two read tokens on our edge config store.",
        "Delete those edge config access tokens, they were shared too widely.",
        "Remove the stale tokens from the edge configuration.",
    ],
    "vercel/createedgeconfig": [
        "Create a new edge config store for feature settings.",
        "Set up an edge configuration named 'pricing-flags'.",
        "I want a fresh edge config under our team scope.",
    ],
    "vercel/deleteconfiguration": [
        "Uninstall the Slack integration configuration from the account.",
        "Remove that integration's configuration by its id.",
        "Delete the marketplace integration setup we no longer use.",
    ],
    "zoom/account": [
        "Show the settings and details of sub account 5512.",
        "Get the profile of that sub account under our master org.",
        "Pull the record for one of our managed sub accounts.",
    ],
    "zoom/accountdisassociate": [
        "Detach the acquired company's sub account from our master account.",
        "Disassociate sub account 7781 so it becomes standalone.",
        "Cut the link between the master org and that sub account.",
    ],
    "zoom/accounts": [
        "List every sub account under our master organization.",
        "Which managed sub accounts exist on this enterprise account?",
        "Show all the sub accounts we've created.",
    ],
}


def main() -> None:
    paths.ensure_dirs()
    catalog = json.loads(
        (paths.DATA_DIR / "catalogs" / "multiserver_catalog.json").read_text(encoding="utf-8")
    )

    missing = [key for key in QUERIES if key not in catalog]
    if missing:
        raise KeyError(f"queries reference tools missing from catalog: {missing}")

    rows = []
    counter = 0
    for tool_key, query_list in QUERIES.items():
        for i, query in enumerate(query_list):
            rows.append(
                {
                    "query_id": f"ms-{counter:04d}",
                    "dataset": "ms",
                    "server": catalog[tool_key]["server"],
                    "tool": tool_key,
                    "scenario_id": f"ms:{tool_key}:s{i:02d}",
                    "origin": "llm_claude_eval",
                    "anchor": query,
                    "positive_schema": json.dumps(catalog[tool_key]["schema"], sort_keys=True),
                }
            )
            counter += 1
    queries_df = pd.DataFrame(rows)
    queries_path = paths.DATA_DIR / "queries_multiserver.csv"
    queries_df.to_csv(queries_path, index=False)
    print(f"wrote {queries_path}: {len(queries_df)} queries over {queries_df['tool'].nunique()} tools "
          f"across {queries_df['server'].nunique()} servers")

    # Merged corpus: 30 GitHub tools (plain-name keys) + all multi-server tools.
    github_corpus = json.loads(paths.CORPUS_JSON.read_text(encoding="utf-8"))
    merged: dict[str, dict] = {}
    for tool, entry in github_corpus.items():
        merged[tool] = {"tool": tool, "server": entry["server"], "schema": entry["schema"]}
    for tool_key, entry in catalog.items():
        merged[tool_key] = {"tool": tool_key, "server": entry["server"], "schema": entry["schema"]}
    merged_path = paths.DATA_DIR / "corpus_multiserver.json"
    merged_path.write_text(json.dumps(merged, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {merged_path}: {len(merged)} tools")

    # Regime 3: train/val from regime 1 (GitHub-only), test = all ms queries,
    # ranked against the merged corpus.
    regime1 = json.loads(
        (paths.SPLITS_DIR / "regime1_unseen_queries.json").read_text(encoding="utf-8")
    )
    regime3 = {
        "regime": "regime3_unseen_servers",
        "seed": regime1["seed"],
        "corpus_file": "corpus_multiserver.json",
        "queries_files": ["queries_multiserver.csv"],
        "corpus_tools": sorted(merged),
        "train": regime1["train"],
        "val": regime1["val"],
        "test": queries_df["query_id"].tolist(),
    }
    split_path = paths.SPLITS_DIR / "regime3_unseen_servers.json"
    split_path.write_text(json.dumps(regime3, indent=1), encoding="utf-8")
    print(f"wrote {split_path}: test={len(regime3['test'])} corpus={len(regime3['corpus_tools'])}")


if __name__ == "__main__":
    main()
