use super::*;
use jcode_provider_core::{RouteCheapnessEstimate, RouteCostConfidence, RouteCostSource};

const POLICY: &str = include_str!("best_fit_router_fixture.json");

fn route(
    model: &str,
    api: &str,
    detail: &str,
    kind: Option<RouteBillingKind>,
    cost: u64,
) -> ModelRoute {
    ModelRoute {
        model: model.into(),
        provider: "p".into(),
        api_method: api.into(),
        available: true,
        detail: detail.into(),
        cheapness: kind.map(|billing_kind| RouteCheapnessEstimate {
            billing_kind,
            source: RouteCostSource::PublicApiPricing,
            confidence: RouteCostConfidence::High,
            monthly_price_micros: None,
            input_price_per_mtok_micros: None,
            output_price_per_mtok_micros: None,
            cache_read_price_per_mtok_micros: None,
            included_requests_per_month: None,
            reference_input_tokens: 0,
            reference_output_tokens: 0,
            estimated_reference_cost_micros: Some(cost),
            note: None,
        }),
        usage: None,
    }
}

fn routes() -> Vec<ModelRoute> {
    vec![
        route("m-metered", "x-api", "", Some(RouteBillingKind::Metered), 9),
        route("m-web", "x-web", "", None, 0),
        route(
            "m-oauth-big",
            "x-oauth@a",
            "",
            Some(RouteBillingKind::Subscription),
            50,
        ),
        route(
            "m-oauth-small",
            "x-oauth@a",
            "",
            Some(RouteBillingKind::Subscription),
            5,
        ),
        route(
            "m-local",
            "openai-compatible:loc",
            "(http://localhost:1/v1)",
            None,
            0,
        ),
        route(
            "m-approved",
            "openai-compatible:paid",
            "(https://h/v1)",
            None,
            1,
        ),
        route("m-notools", "x-oauth@b", "no tools", None, 0),
    ]
}

fn approved() -> HashSet<String> {
    ["paid".to_string()].into()
}

#[test]
fn blocks_unapproved_metered_and_browser() {
    let p = Policy::from_json(POLICY).unwrap();
    let r = routes();
    assert_eq!(p.classify(&r[0], &approved()), Billing::Metered);
    assert!(!p.allowed(Billing::Metered));
    assert_eq!(p.classify(&r[1], &approved()), Billing::Browser);
    assert!(!p.allowed(Billing::Browser));
    assert_eq!(p.classify(&r[5], &HashSet::new()), Billing::Unknown);
    assert_eq!(p.classify(&r[5], &approved()), Billing::MeteredApproved);
    assert!(p.allowed(Billing::MeteredApproved));
}

#[test]
fn deep_tier_prefers_oauth_highest_reasoning_with_tools() {
    let p = Policy::from_json(POLICY).unwrap();
    let r = routes();
    let ranked = p.rank(PARENT_TIER, &r, &approved(), 0, &HashSet::new(), &|_| true);
    assert_eq!(ranked[0].0.model, "m-oauth-big");
    assert!(
        ranked
            .iter()
            .all(|(r, _)| r.model != "m-notools" && r.model != "m-metered" && r.model != "m-web")
    );
}

#[test]
fn shard_tier_conserves_quota_and_local_needs_probe() {
    let p = Policy::from_json(POLICY).unwrap();
    let r = routes();
    let warm = p.rank(SWARM_TIER, &r, &approved(), 0, &HashSet::new(), &|_| true);
    assert_eq!(warm[0].0.model, "m-local");
    assert_eq!(warm[1].0.model, "m-approved");
    let cold = p.rank(SWARM_TIER, &r, &approved(), 0, &HashSet::new(), &|_| false);
    assert_eq!(cold[0].0.model, "m-approved");
}

#[test]
fn sticky_pin_then_step_down() {
    let p = Policy::from_json(POLICY).unwrap();
    let r = routes();
    let (a, _, sticky_a) = pick(&p, "s-test", PARENT_TIER, &r, &|_| true).unwrap();
    let (b, _, sticky_b) = pick(&p, "s-test", PARENT_TIER, &r, &|_| true).unwrap();
    assert_eq!(
        (a.model.as_str(), sticky_a, sticky_b),
        ("m-oauth-big", false, true)
    );
    assert_eq!(b.model, a.model);
    assert!(!step_down(&p, "s-test", PARENT_TIER, "server_error"));
    assert!(step_down(
        &p,
        "s-test",
        PARENT_TIER,
        step_down_reason("HTTP 429 Too Many Requests").unwrap()
    ));
    let (c, _, sticky_c) = pick(&p, "s-test", PARENT_TIER, &r, &|_| true).unwrap();
    assert_eq!((c.model.as_str(), sticky_c), ("m-oauth-small", false));
}

#[test]
fn local_probe_rejects_closed_port_fast() {
    let r = route(
        "m",
        "openai-compatible:x",
        "(http://127.0.0.1:1/v1)",
        None,
        0,
    );
    let t = Instant::now();
    assert!(!local_probe(&r));
    assert!(t.elapsed() < Duration::from_secs(3));
}

#[test]
fn metered_approval_reads_provider_toml() {
    let got = approved_from_toml(
        "[providers.Paid]\nmetered_approved = true\n[providers.other]\nbase_url = \"x\"\n",
    );
    assert_eq!(got, ["paid".to_string()].into());
}

fn ctx_route(model: &str, api: &str, detail: &str, cost: u64) -> ModelRoute {
    route(
        model,
        api,
        detail,
        Some(RouteBillingKind::Subscription),
        cost,
    )
}

fn with_tmp_state<T>(f: impl FnOnce() -> T) -> T {
    static L: Mutex<()> = Mutex::new(());
    let _g = L.lock().unwrap_or_else(|e| e.into_inner());
    let dir = std::env::temp_dir().join(format!("bfr-{}-{:?}", std::process::id(), Instant::now()));
    std::fs::create_dir_all(dir.join("logs")).unwrap();
    // JCODE_HOME too: telemetry() writes `<jcode_dir>/logs/hook-telemetry.jsonl`.
    let prev_home = std::env::var_os("JCODE_HOME");
    unsafe { std::env::set_var("JCODE_ROUTER_STATE", dir.join("router-state.json")) };
    unsafe { std::env::set_var("JCODE_HOME", &dir) };
    let out = f();
    unsafe { std::env::remove_var("JCODE_ROUTER_STATE") };
    match prev_home {
        Some(v) => unsafe { std::env::set_var("JCODE_HOME", v) },
        None => unsafe { std::env::remove_var("JCODE_HOME") },
    }
    let _ = std::fs::remove_dir_all(dir);
    out
}

#[test]
fn classifies_spawns_into_tiers() {
    let p = Policy::from_json(POLICY).unwrap();
    assert_eq!(
        p.classify_spawn("Implement the parser and add tests"),
        CODE_TIER
    );
    assert_eq!(
        p.classify_spawn("Read src/a.rs and src/b.rs, report findings"),
        LOCAL_TIER
    );
    assert_eq!(
        p.classify_spawn("Audit the whole repo for unsafe blocks"),
        SWARM_TIER
    );
    assert_eq!(
        p.classify_spawn("Read a.rs b.rs c.rs d.rs and summarize"),
        SWARM_TIER
    );
}

#[test]
fn true_local_rejects_cloud_and_proxy_tags() {
    let p = Policy::from_json(POLICY).unwrap();
    assert!(p.is_true_local(&route(
        "m",
        "openai-compatible:x",
        "(http://127.0.0.1:11434/v1)",
        None,
        0
    )));
    assert!(!p.is_true_local(&route(
        "m:cloud",
        "openai-compatible:x",
        "(http://localhost:11434/v1)",
        None,
        0
    )));
    assert!(!p.is_true_local(&route(
        "m",
        "openai-compatible:x",
        "(http://localhost:8080/v1) proxy",
        None,
        0
    )));
    assert!(!p.is_true_local(&route(
        "m",
        "openai-compatible:x",
        "(https://remote.example/v1)",
        None,
        0
    )));
}

#[test]
fn context_floor_and_payload_escalation() {
    let p = Policy::from_json(POLICY).unwrap();
    let r = vec![
        ctx_route("small", "x-oauth@a", "context ~70000 tokens", 50),
        ctx_route("big", "x-oauth@a", "context ~200000 tokens", 10),
    ];
    let code = p.rank(CODE_TIER, &r, &HashSet::new(), 0, &HashSet::new(), &|_| {
        true
    });
    assert_eq!(code.len(), 2);
    assert_eq!(
        p.rank(
            PARENT_TIER,
            &r,
            &HashSet::new(),
            0,
            &HashSet::new(),
            &|_| true
        )[0]
        .0
        .model,
        "big"
    );
    // 50k tokens * 1.5 = 75k > 70k: only the big window survives.
    let esc = p.rank(
        CODE_TIER,
        &r,
        &HashSet::new(),
        50_000,
        &HashSet::new(),
        &|_| true,
    );
    assert_eq!(
        esc.iter().map(|x| x.0.model.as_str()).collect::<Vec<_>>(),
        ["big"]
    );
}

#[test]
fn breaker_trips_only_on_listed_reasons_and_excludes_route() {
    with_tmp_state(|| {
        let p = Policy::from_json(POLICY).unwrap();
        let r = vec![
            ctx_route("a", "x-oauth@a", "context ~200000 tokens", 50),
            ctx_route("b", "x-oauth@a", "context ~200000 tokens", 5),
        ];
        assert!(!p.trip("a@x-oauth@a", "cap_exhausted"));
        assert!(p.trip("a@x-oauth@a", "429"));
        let st = with_state(|s| s.breakers.clone()).unwrap();
        let until = st["a@x-oauth@a"];
        assert!(until >= now_s() + 119 && until <= now_s() + 121);
        let d = dispatch(
            &p,
            "s-brk",
            "implement x",
            &r,
            &|_| true,
            std::process::id(),
        )
        .unwrap();
        assert_eq!((d.route.model.as_str(), d.tier), ("b", CODE_TIER));
    });
}

#[test]
fn local_slot_limit_spills_and_stale_pid_is_cleaned() {
    with_tmp_state(|| {
        let p = Policy::from_json(POLICY).unwrap();
        let r = vec![
            route(
                "loc",
                "openai-compatible:l",
                "(http://127.0.0.1:9/v1) context ~32000 tokens",
                None,
                0,
            ),
            ctx_route("shard", "x-oauth@a", "context ~200000 tokens", 1),
        ];
        // Another live process (pid 1) holds the only local slot -> spill to parallel_shard.
        with_state(|s| {
            s.local_slots
                .insert("loc@openai-compatible:l".into(), (1, now_s() + 600))
        });
        let d = dispatch(&p, "s-loc", "read a.rs", &r, &|_| true, std::process::id()).unwrap();
        assert_eq!(
            (d.requested_tier, d.tier, d.route.model.as_str()),
            (LOCAL_TIER, SWARM_TIER, "shard")
        );
        // A dead holder pid is reaped, so the local slot is ours.
        with_state(|s| {
            s.local_slots
                .insert("loc@openai-compatible:l".into(), (9_999_999, now_s() + 600))
        });
        let d = dispatch(&p, "s-loc2", "read a.rs", &r, &|_| true, std::process::id()).unwrap();
        assert_eq!((d.tier, d.billing), (LOCAL_TIER, Billing::Local));
    });
}

#[test]
fn metered_budget_steps_down_to_zero_marginal_cost() {
    with_tmp_state(|| {
        let p = Policy::from_json(POLICY).unwrap();
        let mut paid = route(
            "paid",
            "openai-compatible:paid",
            "(https://h/v1) context ~200000 tokens",
            None,
            1,
        );
        paid.cheapness = None;
        let r = vec![
            paid,
            ctx_route("sub", "x-oauth@a", "context ~200000 tokens", 9),
        ];
        let approved: HashSet<String> = ["paid".to_string()].into();
        // parallel_shard prefers metered_approved after local.
        assert_eq!(
            p.rank(SWARM_TIER, &r, &approved, 0, &HashSet::new(), &|_| true)[0]
                .0
                .model,
            "paid"
        );
        with_state(|s| s.metered_tokens.insert("s-bud".into(), u64::MAX / 2));
        // shortcut: dispatch reads approvals from the real config.toml, so assert the
        // budget filter directly on the over-budget allow-list.
        assert!(!p.over_budget_allowed(Billing::MeteredApproved));
        assert!(p.over_budget_allowed(Billing::Local) && p.over_budget_allowed(Billing::Oauth));
        let d = dispatch(
            &p,
            "s-bud",
            "audit everything",
            &r,
            &|_| true,
            std::process::id(),
        )
        .unwrap();
        assert!(d.over_budget && d.billing != Billing::MeteredApproved);
    });
}

#[test]
fn env_override_reads_tier_variable() {
    let p = Policy::from_json(POLICY).unwrap();
    unsafe { std::env::set_var("JCODE_ROUTE_TIER_MODEL_CODE_EXECUTION", "m-x") };
    assert_eq!(p.env_override(CODE_TIER).as_deref(), Some("m-x"));
    unsafe { std::env::remove_var("JCODE_ROUTE_TIER_MODEL_CODE_EXECUTION") };
    assert_eq!(p.env_override(CODE_TIER), None);
}

#[test]
fn corrupt_router_state_resets_instead_of_failing() {
    with_tmp_state(|| {
        let path = state_path().unwrap();
        std::fs::write(&path, "{not json").unwrap();
        assert_eq!(with_state(|s| s.breakers.len()), Some(0));
        let p = Policy::from_json(POLICY).unwrap();
        assert!(p.trip("r", "timeout"));
        assert_eq!(with_state(|s| s.breakers.contains_key("r")), Some(true));
    });
}

#[test]
fn shared_state_lock_serializes_concurrent_writers() {
    with_tmp_state(|| {
        let hs: Vec<_> = (0..8)
            .map(|_| {
                let p = std::env::var("JCODE_ROUTER_STATE").unwrap();
                std::thread::spawn(move || {
                    unsafe { std::env::set_var("JCODE_ROUTER_STATE", p) };
                    with_state(|s| *s.metered_tokens.entry("c".into()).or_default() += 1)
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        assert_eq!(with_state(|s| s.metered_tokens["c"]), Some(8));
    });
}

#[test]
fn live_shaped_routes_without_metadata_resolve_every_tier() {
    // Live list_models rows: no cheapness, no advertised window.
    let p = Policy::from_json(POLICY).unwrap();
    let r = vec![
        route("m-oauth", "code-assist-oauth", "", None, 0),
        route("m-acct", "claude-oauth@acct", "account acct", None, 0),
        route(
            "m-loc",
            "openai-compatible:l",
            "http://localhost:11434/v1",
            None,
            0,
        ),
    ];
    for tier in ALL_TIERS {
        let ranked = p.rank(tier, &r, &HashSet::new(), 1_000, &HashSet::new(), &|_| true);
        assert!(!ranked.is_empty(), "{tier} resolved nothing");
    }
    let cold = p.rank(SWARM_TIER, &r, &HashSet::new(), 0, &HashSet::new(), &|_| {
        false
    });
    assert_eq!(cold[0].1, Billing::Oauth);
}

#[test]
fn local_probe_tries_every_resolved_address() {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    // `localhost` may resolve to ::1 first; the IPv4 listener must still count.
    let r = route(
        "m",
        "openai-compatible:x",
        &format!("http://localhost:{port}/v1"),
        None,
        0,
    );
    assert!(local_probe(&r));
}

#[test]
fn local_slot_released_on_worker_exit() {
    with_tmp_state(|| {
        let p = Policy::from_json(POLICY).unwrap();
        let r = vec![route(
            "loc",
            "openai-compatible:l",
            "(http://127.0.0.1:9/v1)",
            None,
            0,
        )];
        let d = dispatch(
            &p,
            "coord-rel",
            "read a.rs",
            &r,
            &|_| true,
            std::process::id(),
        )
        .unwrap();
        assert_eq!(d.billing, Billing::Local);
        assert_eq!(with_state(|s| s.local_slots.len()), Some(1));
        bind_local_slot("coord-rel", "worker-rel");
        release_local_slot("worker-rel");
        assert_eq!(with_state(|s| s.local_slots.len()), Some(0));
        release_local_slot("worker-rel"); // idempotent
    });
}

#[test]
fn manual_pin_beats_tier_until_breaker_then_reports_fallback() {
    with_tmp_state(|| {
        let p = Policy::from_json(POLICY).unwrap();
        assert!(manual_pin("s-pin").is_none());
        assert_eq!(
            selection_status("s-pin", "a:x", "parent"),
            "Auto (parent): a:x"
        );
        set_manual_pin("s-pin", "a:x", "x@a");
        let (spec, route, until) = manual_pin("s-pin").unwrap();
        assert_eq!((spec.as_str(), route.as_str(), until), ("a:x", "x@a", None));
        assert_eq!(selection_status("s-pin", "a:x", "parent"), "Pinned: a:x");
        // Throttle opens the breaker; status names the fallback and reset time.
        let until = throttle_pin(&p, "x@a").unwrap();
        assert_eq!(manual_pin("s-pin").unwrap().2, Some(until));
        let s = selection_status("s-pin", "b:y", "parent");
        assert!(
            s.starts_with("Throttled (a:x, resets ") && s.ends_with(") -> Fallback: b:y"),
            "{s}"
        );
        // Expired breaker: pin is live again (auto-restore precondition).
        with_state(|st| st.breakers.insert("x@a".into(), 1));
        assert_eq!(manual_pin("s-pin").unwrap().2, None);
    });
}

#[test]
fn throttle_pin_keeps_existing_cooldown_and_status_survives_bad_timestamp() {
    with_tmp_state(|| {
        let p = Policy::from_json(POLICY).unwrap();
        set_manual_pin("s-t", "a:x", "x@a");
        let first = throttle_pin(&p, "x@a").unwrap();
        // A second throttle must not extend the window (restore stays predictable).
        assert_eq!(throttle_pin(&p, "x@a"), Some(first));
        with_state(|st| st.breakers.insert("x@a".into(), i64::MAX as u64));
        let s = selection_status("s-t", "b:y", "parent");
        assert_eq!(s, "Throttled (a:x, resets ?) -> Fallback: b:y");
    });
}

#[test]
fn manual_pin_survives_tier_reclassification_and_records_triggers() {
    with_tmp_state(|| {
        let p = Policy::from_json(POLICY).unwrap();
        let r = vec![
            ctx_route("big", "x-oauth@a", "context ~200000 tokens", 50),
            ctx_route("small", "x-oauth@a", "context ~200000 tokens", 5),
        ];
        set_manual_pin("s-tier", "a:small", "small@x-oauth@a");
        record_trigger("s-tier", "manual_pin", "a:small");
        // code_execution and deep_architecture prompts both land on the pin.
        for prompt in [
            "implement the parser and add tests",
            "design the architecture",
        ] {
            let d = dispatch(&p, "s-tier", prompt, &r, &|_| true, std::process::id()).unwrap();
            assert_eq!(
                (d.route.model.as_str(), d.sticky),
                ("small", true),
                "{prompt}"
            );
        }
        assert_eq!(
            last_trigger("s-tier"),
            Some(("manual_pin".into(), "a:small".into()))
        );
        // Throttle: the breaker opens, the pin yields to ranking, the trigger is recorded.
        let until = throttle_pin(&p, "small@x-oauth@a").unwrap();
        assert!(until >= now_s() + 119 && until <= now_s() + 121);
        let d = dispatch(
            &p,
            "s-tier",
            "implement x",
            &r,
            &|_| true,
            std::process::id(),
        )
        .unwrap();
        assert_eq!(d.route.model, "big");
        record_trigger("s-tier", "rate_limit_429", "a:big");
        assert_eq!(last_trigger("s-tier").unwrap().0, "rate_limit_429");
        // Cooldown expiry restores the pin on the next dispatch.
        with_state(|st| st.breakers.insert("small@x-oauth@a".into(), 1));
        let d = dispatch(
            &p,
            "s-tier",
            "implement x",
            &r,
            &|_| true,
            std::process::id(),
        )
        .unwrap();
        assert_eq!(d.route.model, "small");
        record_trigger("s-tier", "cooldown_recovered", "a:small");
        assert_eq!(last_trigger("s-tier").unwrap().0, "cooldown_recovered");
        let rows = with_state(|st| st.last_trigger.len()).unwrap();
        assert_eq!(rows, 1, "one row per session, overwritten on each switch");
    });
}

#[test]
fn step_down_exclusion_expires_after_ttl() {
    let p = Policy::from_json(POLICY).unwrap();
    let r = routes();
    let (a, _, _) = pick(&p, "s-ttl", PARENT_TIER, &r, &|_| true).unwrap();
    assert_eq!(a.model, "m-oauth-big");
    assert!(step_down(&p, "s-ttl", PARENT_TIER, "429"));
    let (b, _, _) = pick(&p, "s-ttl", PARENT_TIER, &r, &|_| true).unwrap();
    assert_eq!(b.model, "m-oauth-small");
    // Expire the exclusion: the stepped-down route is eligible again.
    {
        let mut m = pins().lock().unwrap();
        let e = m
            .get_mut(&("s-ttl".to_string(), PARENT_TIER.to_string()))
            .unwrap();
        for v in e.1.values_mut() {
            *v = 1;
        }
        e.0.clear();
    }
    let (c, _, _) = pick(&p, "s-ttl", PARENT_TIER, &r, &|_| true).unwrap();
    assert_eq!(c.model, "m-oauth-big");
}

#[test]
fn breaker_backoff_doubles_and_caps() {
    with_tmp_state(|| {
        let p = Policy::from_json(POLICY).unwrap();
        let t0 = now_s();
        assert!(p.trip("r@x", "429"));
        let u1 = with_state(|s| s.breakers["r@x"]).unwrap();
        assert!((119..=121).contains(&(u1 - t0)), "first trip: cooldown_s");
        with_state(|s| s.breakers.remove("r@x"));
        assert!(p.trip("r@x", "429"));
        let u2 = with_state(|s| s.breakers["r@x"]).unwrap();
        assert!(
            (239..=241).contains(&(u2 - t0)),
            "second trip: x2, got {}",
            u2 - t0
        );
        for _ in 0..6 {
            with_state(|s| s.breakers.remove("r@x"));
            assert!(p.trip("r@x", "429"));
        }
        let un = with_state(|s| s.breakers["r@x"]).unwrap();
        assert!(
            (1799..=1801).contains(&(un - t0)),
            "capped at max_cooldown_s, got {}",
            un - t0
        );
        // A trip older than the window resets the count.
        with_state(|s| {
            s.breakers.remove("r@x");
            s.trips.insert("r@x".into(), (8, t0 - 5000));
        });
        assert!(p.trip("r@x", "429"));
        let ur = with_state(|s| s.breakers["r@x"]).unwrap();
        assert!(
            (119..=121).contains(&(ur - now_s())),
            "reset to base, got {}",
            ur - now_s()
        );
        // throttle_pin never shortens an already-open breaker.
        assert_eq!(throttle_pin(&p, "r@x"), Some(ur));
    });
}

#[test]
fn concurrent_writers_keep_every_pin_and_breaker() {
    with_tmp_state(|| {
        let p = Policy::from_json(POLICY).unwrap();
        let path = state_path().unwrap();
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let p = Policy::from_json(POLICY).unwrap();
                let path = path.clone();
                std::thread::spawn(move || {
                    // Each thread re-reads the env-pinned path through state_path().
                    assert_eq!(state_path().unwrap(), path);
                    for j in 0..25 {
                        set_manual_pin(&format!("s{i}-{j}"), "a:x", &format!("x{i}@a"));
                        p.trip(&format!("x{i}-{j}@a"), "429");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let (pins, breakers) = with_state(|st| (st.manual_pins.len(), st.breakers.len())).unwrap();
        assert_eq!(
            (pins, breakers),
            (200, 200),
            "no lost writes under contention"
        );
        // The file on disk is well-formed JSON (atomic tmp+rename).
        let text = std::fs::read_to_string(&path).unwrap();
        let st: SharedState = serde_json::from_str(&text).unwrap();
        assert_eq!(st.manual_pins.len(), 200);
        // One session's 429 never evicts another session's pin.
        assert!(p.trip("x0@a", "429"));
        assert_eq!(manual_pin("s1-0").unwrap().2, None);
        assert!(manual_pin("s0-0").unwrap().2.is_some());
    });
}

#[test]
fn state_path_and_telemetry_follow_jcode_home_when_no_override() {
    with_tmp_state(|| {
        unsafe { std::env::remove_var("JCODE_ROUTER_STATE") };
        let home = std::path::PathBuf::from(std::env::var_os("JCODE_HOME").unwrap());
        assert_eq!(state_path().unwrap(), home.join("state/router-state.json"));
        telemetry(serde_json::json!({"event": "probe"}));
        let rows = std::fs::read_to_string(home.join("logs/hook-telemetry.jsonl")).unwrap();
        assert!(rows.contains("\"event\":\"probe\""), "{rows}");
    });
}

#[test]
fn read_only_prompts_never_pick_code_execution() {
    let p = Policy::from_json(POLICY).unwrap();
    for prompt in [
        "Audit src/a.rs for bugs. Do not edit anything.",
        "Read-only: grep the repo for unsafe and report; do not write files",
        "Inspect the prefix handling in src/x.rs",
        "Review how errors are created in a.rs b.rs c.rs d.rs, read only",
    ] {
        let c = p.classify_spawn_detail(prompt);
        assert!(!c.mutating, "{prompt}");
        assert_ne!(c.tier, CODE_TIER, "{prompt}");
    }
    let c = p.classify_spawn_detail("Audit src/a.rs for bugs. Do not edit anything.");
    assert_eq!(c.tier, LOCAL_TIER);
    assert_eq!(c.files, vec!["src/a.rs".to_string()]);
    let m = p.classify_spawn_detail("Fix the bug in src/a.rs");
    assert!(m.mutating);
    assert_eq!(m.tier, CODE_TIER);
    assert_eq!(m.reason, "mutating_marker:fix");
}

#[test]
fn swarm_dispatch_event_has_non_null_reason_mutating_files() {
    let p = Policy::from_json(POLICY).unwrap();
    let c = p.classify_spawn_detail("Do not edit anything. Summarize README.md");
    for picked in [
        None,
        Some(Dispatch {
            route: route("m", "ollama", "(http://127.0.0.1:11434)", None, 0),
            billing: Billing::Local,
            tier: SWARM_TIER,
            requested_tier: c.tier,
            sticky: false,
            over_budget: false,
            payload_tokens: 1,
        }),
    ] {
        let ev = dispatch_event("s", &c, picked.as_ref(), Duration::from_millis(1));
        assert_eq!(ev["event"], "swarm_dispatch");
        assert_eq!(ev["mutating"], false);
        assert!(ev["reason"].as_str().is_some_and(|r| !r.is_empty()));
        assert_eq!(ev["files"], serde_json::json!(["readme.md"]));
    }
}
