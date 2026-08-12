CREATE PROCEDURE [dbo].[HRMS_DoReclock.1.0.1]
	@memberId INT,
	@date DATE,
	@time TIME,
	@reason NVARCHAR(200),
	@location NVARCHAR(200),
	@isClockIn BIT
AS
BEGIN
	SET NOCOUNT ON;

	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK)
		WHERE [MemberId] = @memberId AND [Status] IN (0,1) AND [IsCancel] = 0
		AND CAST(@date AS DATETIME) + CAST(@time AS DATETIME)
		BETWEEN
			-- TAKE LEAVE START FROM MORNING
			CASE WHEN DATEPART(hour, [StartDate]) < 12 
			THEN CAST(CAST([StartDate] AS DATE) AS DATETIME)
			-- TAKE LEAVE START FROM AFTERNOON
			ELSE [StartDate] END
			
			-- TAKE LEAVE END IN THE AFTERNOON
		AND CASE WHEN DATEPART(hour, [EndDate]) > 13
			THEN CONVERT(DATETIME, CONVERT(varchar(10), [EndDate], 120) + ' 23:59:59', 120)
			-- TAKE LEAVE END IN THE MORNING 
			ELSE [EndDate] END)
	BEGIN
		SELECT 29 AS ErrorCode;
		RETURN;
	END

	IF NOT EXISTS (SELECT TOP 1 1 FROM [dbo].[ReClockRecord] WITH(NOLOCK) 
	WHERE [MemberId] = @memberId AND [Date] = @date AND [IsClockIn] = @isClockIn)
	BEGIN
		INSERT INTO [dbo].[ReClockRecord] ([MemberId], [Date], [Time], [IsClockIn], [Reason], [Location])
		OUTPUT 0 AS ErrorCode, INSERTED.RequestId
		VALUES (@memberId, @date, @time, @isClockIn, @reason, @location);
	END
	ELSE
	BEGIN
		SELECT 16 AS ErrorCode;
	END
END
