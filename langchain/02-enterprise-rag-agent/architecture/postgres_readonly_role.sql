-- DBA-only example for a DEDICATED NexusAgent database, after loading the schema.
-- Do not run as agent credentials. This file is never executed by the agent.
-- PUBLIC revocations affect other users of this dedicated database: review them
-- before adapting this example to a shared enterprise database.
BEGIN;
CREATE ROLE nexus_agent_reader LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOINHERIT NOREPLICATION NOBYPASSRLS;
-- Provision its password separately (for example psql \password), never in Git.
-- Grant no role memberships and keep database/schema/table ownership with the DBA.
DO $$
BEGIN
    EXECUTE format('REVOKE CREATE, TEMPORARY ON DATABASE %I FROM PUBLIC', current_database());
    EXECUTE format('REVOKE ALL ON DATABASE %I FROM nexus_agent_reader', current_database());
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO nexus_agent_reader', current_database());
END $$;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO nexus_agent_reader;
REVOKE ALL ON TABLE public.regional_sales, public.employee_roster,
    public.client_contracts FROM PUBLIC, nexus_agent_reader;
GRANT SELECT ON TABLE public.regional_sales, public.employee_roster,
    public.client_contracts TO nexus_agent_reader;
ALTER ROLE nexus_agent_reader SET default_transaction_read_only = on;
ALTER ROLE nexus_agent_reader SET statement_timeout = '15s';
ALTER ROLE nexus_agent_reader SET search_path = pg_catalog, public;
COMMIT;
-- No blanket grants on future tables. Audit future grants, views/functions and
-- trusted schema owners; the runtime fails closed on detected write privileges.
