CREATE PROCEDURE [dbo].[HRMS_CheckClockStatus.1.0.1]
	@memberId INT,
	@date DATE
AS
BEGIN
	SET NOCOUNT ON;

	DECLARE @IsOnLeave AS BIT = 0;
	DECLARE @ClockInStatus AS INT = 5;
	DECLARE @ClockOutStatus AS INT = 5;
    DECLARE @ClockInTime AS TIME;
    DECLARE @ClockOutTime AS TIME;

	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK)
		WHERE [MemberId] = @memberId AND [Status] IN (0,1) AND [IsCancel] = 0
		AND @date BETWEEN CAST([StartDate] AS DATE) AND CAST([EndDate] AS DATE))
	BEGIN
		SET @IsOnLeave = 1;
	END

	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[Attendances] WITH(NOLOCK)
	WHERE [MemberId] = @memberId AND [WorkDate] = @date)
	BEGIN
		IF ((SELECT [ClockIn] FROM [dbo].[Attendances] WITH(NOLOCK)
		WHERE [MemberId] = @memberId AND [WorkDate] = @date) IS NOT NULL)
		BEGIN
			SET @ClockInStatus = 4;
            SET @ClockInTime =(SELECT [ClockIn] FROM [dbo].[Attendances] WITH(NOLOCK)
		        WHERE [MemberId] = @memberId AND [WorkDate] = @date)
		END
		ELSE IF EXISTS (SELECT TOP 1 1 FROM [dbo].[ReClockRecord] WITH(NOLOCK)
		WHERE [MemberId] = @memberId AND [Date] = @date AND [IsClockIn] = 1
		AND [Status] IN (0,1))
		BEGIN
			SET @ClockInStatus = 6;
		END

		IF ((SELECT [ClockOut] FROM [dbo].[Attendances] WITH(NOLOCK)
		WHERE [MemberId] = @memberId AND [WorkDate] = @date) IS NOT NULL)
		BEGIN
			SET @ClockOutStatus = 4;
            SET @ClockOutTime =(SELECT [ClockOut] FROM [dbo].[Attendances] WITH(NOLOCK)
		        WHERE [MemberId] = @memberId AND [WorkDate] = @date)
		END
		ELSE IF EXISTS (SELECT TOP 1 1 FROM [dbo].[ReClockRecord] WITH(NOLOCK)
		WHERE [MemberId] = @memberId AND [Date] = @date AND [IsClockIn] = 0
		AND [Status] IN (0,1))
		BEGIN
			SET @ClockOutStatus = 6;
		END
	END
	ELSE
	BEGIN
		IF EXISTS (SELECT TOP 1 1 FROM [dbo].[ReClockRecord] WITH(NOLOCK)
		WHERE [MemberId] = @memberId AND [Date] = @date AND [IsClockIn] = 1
		AND [Status] IN (0,1))
		BEGIN
			SET @ClockInStatus = 6;
		END
		IF EXISTS (SELECT TOP 1 1 FROM [dbo].[ReClockRecord] WITH(NOLOCK)
		WHERE [MemberId] = @memberId AND [Date] = @date AND [IsClockIn] = 0
		AND [Status] IN (0,1))
		BEGIN
			SET @ClockOutStatus = 6;
		END
	END
	
	SELECT @IsOnLeave AS IsOnLeave, @ClockInStatus AS ClockInStatus, @ClockOutStatus AS ClockOutStatus, @ClockInTime AS ClockInTime, @ClockOutTime AS ClockOutTime;
END