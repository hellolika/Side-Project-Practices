CREATE PROCEDURE [dbo].[HRMS_UpdateTakeLeave.1.0.0]
	@requestId INT = 0,
	@memberId INT,
	@numberOfDay AS DECIMAL(16,9),
	@startDate DATETIME,
	@endDate DATETIME,
	@leaveType INT,
    @image VARCHAR(100),
	@reason NVARCHAR(500)
AS
BEGIN
    SET NOCOUNT ON
	-- CHECKING IF LEAVE REQUEST IS ALREADY EXIST
	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[TakeLeaveRecord] WHERE RequestId = @requestId)
        BEGIN
			DECLARE @oldAmount AS DECIMAL(16,9)
			DECLARE @oldLeaveType AS DECIMAL(16,9)
			DECLARE @year AS DATETIME = YEAR(@startDate)
            DECLARE @defaultLeaveAmount AS DECIMAL(16,9)
            DECLARE @remainingLeave AS DECIMAL(16,9)
            DECLARE @oldReason AS NVARCHAR(500)

            IF((SELECT [IsCancel] FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK) WHERE [RequestId] = @requestId) = 1)
            BEGIN
                SELECT 27 AS ErrorCode;
                RETURN;
            END

            --CHECKING IF THE LEAVE IS ALREADY REJECTED
            --IF((SELECT [Status] FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK) WHERE [RequestId] = @requestId) = 2)
            --BEGIN=
            --    SELECT 22 AS ErrorCode;
            --    RETURN;
            --END

            DECLARE @isLimited AS BIT
	        SELECT @isLimited = [IsLimited] FROM [dbo].[LeaveType] WITH(NOLOCK) WHERE [TypeId] = @leaveType AND [IsEnable] = 1;
            BEGIN
                SELECT @oldReason = [Reason] FROM [dbo].[TakeLeaveRecord] WHERE [RequestId] = @requestId; 
                IF(@isLimited = 1)
                    BEGIN 
                        DECLARE @oldApprovalStatus AS INT

                        SELECT @oldApprovalStatus = [Status]
                        FROM [dbo].[TakeLeaveRecord]
                        WHERE [RequestId] = @requestId

                        SELECT @defaultLeaveAmount = [DefaultLeavesGranted] FROM [dbo].[LeaveType] WHERE [TypeId] = @leaveType
                        IF(@numberOfDay > @defaultLeaveAmount)
                            BEGIN
                                SELECT 23 AS ErrorCode;
                                RETURN;
                            END

                        DECLARE @leaveAmount AS DECIMAL(16,9);
                        SELECT @oldAmount = [NumberOfDay], @oldLeaveType = [LeaveType] FROM [dbo].[TakeLeaveRecord] WHERE [RequestId] = @requestId;
                        SELECT @leaveAmount = [RemainingLeaves] FROM [dbo].[LeaveAmount] WITH(NOLOCK) 
                        WHERE [MemberId] = @memberId AND [LeaveType] = @leaveType
                        AND [Year] = YEAR(@startDate)

                        IF(@oldLeaveType = @leaveType)
                            BEGIN
                                -- CHECKING IF MEMBER STILL HAVE REMAINING LEAVES
                                IF(@leaveAmount IS NULL OR @leaveAmount + @oldAmount - @numberOfDay < 0)
                                    BEGIN
                                        SELECT 23 AS ErrorCode;
                                        RETURN;
                                    END
                            END
                        ELSE
                            BEGIN
                                -- CHECKING IF MEMBER STILL HAVE REMAINING LEAVES
                                IF(@leaveAmount IS NULL OR @leaveAmount - @numberOfDay < 0)
                                    BEGIN
                                        SELECT 23 AS ErrorCode;
                                        RETURN;
                                    END
                            END

                        --WILL REVERT ONLY LEAVE RECORD IS NOT ALREADY REJECTED
                        IF (@oldApprovalStatus != 2)
                            BEGIN
                                -- REVERT OLD LEAVE AMOUNT
                                SELECT @oldAmount = [NumberOfDay], @oldLeaveType = [LeaveType]  FROM [dbo].[TakeLeaveRecord] WHERE [RequestId] = @requestId;

                                UPDATE [dbo].[LeaveAmount]
                                SET [RemainingLeaves] = [RemainingLeaves] + @oldAmount
                                WHERE [MemberId] = @memberId 
                                AND [Year] = @year AND [LeaveType] = @oldLeaveType;
                            END
                            

                        --UPDATE LEAVE RECORD WITH THE NEW RECORD
                        UPDATE [dbo].[TakeLeaveRecord] 
                        SET [NumberOfDay] = @numberOfDay,
                        [StartDate] = @startDate,
                        [EndDate] = @endDate,
                        [LeaveType] = @leaveType,
                        [Image] = @image,
                        [Reason] = CASE WHEN @reason = '' THEN @oldReason ELSE @reason END
                        WHERE [RequestId] = @requestId
                        AND [IsCancel] = 0;

                        SELECT @remainingLeave = [RemainingLeaves] FROM [dbo].[LeaveAmount] WITH(NOLOCK)
                        WHERE [MemberId] = @memberId AND [LeaveType] = @leaveType
                        AND [Year] = YEAR(@startDate);

                        -- CHECKING IF MEMBER STILL HAVE REMAINING LEAVES
                        -- IF(@remainingLeave IS NULL OR @remainingLeave - @numberOfDay < 0)
                        --     BEGIN
                        --         SELECT 23 AS ErrorCode;
                        --         RETURN;
                        --     END

                        --UPDATE LEAVE AMOUNT ONLY LEAVE RECORD IS NOT ALREADY REJECTED
                        IF (@oldApprovalStatus != 2)
                            BEGIN
                                --UPDATE LEAVE AMOUNT FOR MEMBER
                                UPDATE [dbo].[LeaveAmount] 
                                SET [RemainingLeaves] = [RemainingLeaves] - @numberOfDay  
                                WHERE [MemberId] = @memberId AND [LeaveType] = @leaveType;
                            END

                        SELECT 0 AS ErrorCode, @requestId AS RequestId;
                        RETURN;
                    END
                ELSE
                    BEGIN
                    -- REVERT OLD LEAVE AMOUNT
                    SELECT @oldAmount = [NumberOfDay], @oldLeaveType = [LeaveType]  FROM [dbo].[TakeLeaveRecord] WHERE [RequestId] = @requestId;

                    UPDATE [dbo].[LeaveAmount]
                    SET [RemainingLeaves] = [RemainingLeaves] + @oldAmount
                    WHERE [MemberId] = @memberId 
                    AND [Year] = @year AND [LeaveType] = @oldLeaveType;

                    UPDATE [dbo].[TakeLeaveRecord] 
                    SET [NumberOfDay] = @numberOfDay,
                    [StartDate] = @startDate,
                    [EndDate] = @endDate,
                    [LeaveType] = @leaveType,
                    [Image] = @image,
                    [Reason] = CASE WHEN @reason = '' THEN @oldReason ELSE @reason END
                    WHERE [RequestId] = @requestId
                    AND [IsCancel] = 0;

                    SELECT 0 AS ErrorCode, @requestId AS RequestId; 
                    END    
            END
		END
	ELSE
	BEGIN
		SELECT 1 AS ErrorCode;
		RETURN;
	END
END