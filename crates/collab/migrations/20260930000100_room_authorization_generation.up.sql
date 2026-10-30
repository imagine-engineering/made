ALTER TABLE public.room_participants
    ADD COLUMN authorization_generation bigint NOT NULL DEFAULT 1
        CHECK (authorization_generation > 0);
